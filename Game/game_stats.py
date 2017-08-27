# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 22:59:23 2017

@author: lilin
"""

class GameStats():
    """跟踪游戏的统计信息"""
    def __init__(self,ai_settings):
        """初始化统计信息"""
        self.ai_settings=ai_settings
        self.reset_stats()
        
        #让游戏一开始就处于非活动状态
        self.game_active = False
        
        #游戏刚启动是处于活动状态
#        self.game_active = True

        #在任何情况下都不应重置最高分
        self.high_score = 0
    
    def reset_stats(self):
        """初始化在游戏运行期间可能变化的统计信息"""
        self.ships_left = self.ai_settings.ship_limit
        self.score = 0
        self.level = 1
        
        