# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 21:47:04 2023

@author: Arya
"""

import Constants as const

class Index:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Pixel:
    def __init__(self, index: Index, g_cost=float('inf'), h_cost=float('inf')):
        self.index = index
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = self.g_cost + self.h_cost
        self.parent = None

    def __str__(self):
        return "Index: {0} {1}, Parent: {2}".format(self.index.x, self.index.y, self.parent)

    def update_cost(self, parent: 'Pixel', end: 'Pixel'):
        if (parent.index.x == self.index.x and parent.index.y == self.index.y):
            self.g_cost = parent.g_cost + const.WEIGHT

        dx = abs(parent.index.x - end.index.x)
        dy = abs(parent.index.y - end.index.y)

        self.h_cost = const.WEIGHT * (dx + dy) 
        self.f_cost = self.g_cost + self.h_cost
        self.parent = parent

    @staticmethod
    def compare_cost(curr: 'Pixel', prev: 'Pixel'):
        if curr.f_cost < prev.f_cost:
            return 1
        elif curr.f_cost == prev.f_cost:
            if curr.h_cost <= prev.h_cost:
                return 1
        else:
            return 0