import asyncio
import json
import logging
from typing import List, Dict
import os
import concurrent.futures
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from aiohttp import web

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# download path: https://huggingface.co/BAAI/bge-reranker-v2-m3
default_model = "/data1/r1/bge-reranker-v2-m3"
# 并发数， 按需调整
os.environ["RERANK_MAX_WORKERS"] = "4"

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_lock = asyncio.Lock()
        self.max_workers = int(os.getenv("RERANK_MAX_WORKERS", "4"))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        # 用于限流单请求内的并发评分任务数量
        self._sem = asyncio.Semaphore(self.max_workers)
        logger.info(f"initialize model use device: {self.device}, max_workers={self.max_workers}")

    def _load_model_sync(self):
        """同步加载，放在线程池内执行"""
        logger.info(f"load model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("load complete")

    async def load_model(self):
        """异步加载，带锁防止并发重复加载"""
        if self.model is None:
            async with self._load_lock:
                if self.model is None:
                    try:
                        await asyncio.get_running_loop().run_in_executor(self._executor, self._load_model_sync)
                    except Exception as e:
                        logger.error(f"load failed: {e}")
                        raise

    def _compute_score_sync(self, query: str, passage: str) -> float:
        """同步推理函数，放在线程池执行，避免阻塞事件循环"""
        try:
            inputs = self.tokenizer(
                query,
                passage,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)

            with torch.inference_mode():
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
                scores = torch.sigmoid(scores)
            return float(scores.cpu().numpy()[0])
        except Exception as e:
            logger.error(f"compute score error: {e}")
            return 0.0

    async def compute_score(self, query: str, passage: str) -> float:
        """异步包装：在线程池执行同步推理"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._compute_score_sync, query, passage)

    async def rerank_documents(self, query: str, documents: List[str], top_n: int = None, threshold=0.1) -> List[Dict]:
        """并发重排：对每个文档的打分在线程池并发执行，受信号量限制"""
        if not documents:
            return []
        await self.load_model()
        async def bound_score(doc: str) -> float:
            async with self._sem:
                return await self.compute_score(query, doc)

        # 并行计算各文档分数
        tasks = [asyncio.create_task(bound_score(doc)) for doc in documents]
        scores = await asyncio.gather(*tasks, return_exceptions=False)

        results = [{"index": idx, "relevance_score": float(score)} for idx, score in enumerate(scores)]
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.info(f"{[ite['relevance_score'] for ite in results]}")
        # 按需过滤
        # results = [ite for ite in results if ite['relevance_score'] >= threshold]
        if top_n is not None:
            results = results[:top_n]
        logger.debug(results)
        logger.info(f"rerank complete,return: {len(results)} result")
        return results

bge_reranker = BGEReranker(default_model)

async def handle_rerank(request):
    """util rerank"""
    try:
        start_time = asyncio.get_running_loop().time()
        data = await request.json()
        query = data.get("query", "")
        documents = data.get("documents", [])
        top_n = data.get("top_n")
        model = data.get("model", default_model)
        threshold = data.get("threshold", 0.1)
        if not query or not documents:
            return web.json_response(
                {"error": "query and documents is necessary"},
                status=400
            )
        logger.info(f"rerank query: query='{query[:50]}...', docs={len(documents)}, top_n={top_n}, threshold= {threshold}, model={model}")
        results = await bge_reranker.rerank_documents(query, documents, top_n, threshold)
        response_data = {
            "results": results,
            "model": model,
            "usage": {
                "total_tokens": len(query.split()) + sum(len(doc.split()) for doc in documents)
            }
        }
        cost_time = asyncio.get_running_loop().time() - start_time
        logger.info(f"rerank cost time: {cost_time:.4f}s")
        return web.json_response(response_data)

    except json.JSONDecodeError:
        return web.json_response({"error": "invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"query error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_health(request):
    """health check"""
    model_loaded = bge_reranker.model is not None
    return web.json_response({
        "status": "healthy",
        "service": "bge-rerank",
        "model": bge_reranker.model_name,
        "model_loaded": model_loaded,
        "device": str(bge_reranker.device)
    })

async def handle_models(request):
    """get models list"""
    return web.json_response({
        "models": [
            {
                "id": default_model,
                "description": "BGE-M3 multilingual reranker",
                "loaded": bge_reranker.model is not None
            }
        ]
    })

async def create_app():
    """create web app"""
    app = web.Application()
    # add route
    app.router.add_post("/rerank", handle_rerank)
    app.router.add_post("/v1/rerank", handle_rerank)  # v1 API
    app.router.add_get("/health", handle_health)
    app.router.add_get("/models", handle_models)
    return app

async def main(host, port):
    """start server"""
    app = await create_app()
    logger.info(f"rerank server run at: http://{host}:{port}")
    logger.info("API port:")
    logger.info("  POST /rerank - rerank API")
    logger.info("  POST /v1/rerank - rerank API (v1)")
    logger.info("  GET /health - health check")
    logger.info("  GET /models - model list")

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    print(f"BGE rerank server start at: http://{host}:{port}")
    print("model will load when first request arrives.")
    print(" Ctrl+C stop server.")
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("stop server...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8182
    asyncio.run(main(host, port))