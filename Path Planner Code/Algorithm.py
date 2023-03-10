# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:29:38 2023

@author: Arya
"""

import numpy as np
from matplotlib import pyplot as plt
import Constants as const
from Pixel import Pixel, Index

class A_Star_Algo:
    def __init__(self, data):
        self.data = data
        (self.start, self.end) = self.get_sources()
        self.open_list = [Pixel(index=self.start,g_cost=0,h_cost=0)]
        self.closed_list: list[Pixel] = []

    def get_sources(self):
        sources = np.where(self.data == const.SOURCE)
        print(sources)
        print(self.data)
        if not len(sources[0]) == 2:
            raise Exception("Does not have exactly two sources")
        return (Index(sources[0][0], sources[1][0]), Index(sources[0][1], sources[1][1]))
        # return ((2,2), (50, 50))
    # Pseudocode
    # add start_node to open_list
    # loop:
    #     curr_node = node in open_list with lowest f_value
    #     if curr_node = end:
    #         return
    #     for each neighbor of curr_node:
    #         if neighbor is OBSTACLE or neighbor is in closed_list:
    #             skip to next neighbor
            
    #         if new path to neighbor is shorter or neighbor is in closed_list:
    #             update f_cost of neighbor
    #             set parent of neighbor as current
    #             if neighbor not in open_list:
    #                 add neighbor to open_list
                    
    def algo_exec(self):
        plt.ion()
        while len(self.open_list)!=0:
            temp=0
            for i, node in enumerate(self.open_list):
                if Pixel.compare_cost(node, self.open_list[temp]):
                    temp = i
                    
            curr_node = self.open_list.pop(temp)
            x_curr = curr_node.index.x
            y_curr = curr_node.index.y
            
            neighbors=[Pixel(Index(x_curr+1, y_curr)), Pixel(Index(x_curr-1, y_curr)),Pixel(Index(x_curr, y_curr+1)),Pixel(Index(x_curr, y_curr-1))]
            neighbors_updated = []
            
            for i, node in enumerate(neighbors):
                if node.index.x==self.end.x and node.index.y==self.end.y:
                    plt.close()
                    path_points = self.get_path(curr_node)
                    protocol = self.return_protocol(path_points)
                    return protocol
                if x_curr>=0 and y_curr>=0:
                    node.update_cost(parent = curr_node, end = Pixel(self.end))
                    neighbors_updated.append(node)
                    
            # for node in neighbors_updated:
            #     print (node.index.x, node.index.y)
                    
            for neighbor_index, neighbor_node in enumerate(neighbors_updated):
                poplist = False
                
                for open_index, open_node in enumerate(self.open_list):
                    if open_node.index.x == neighbor_node.index.x and open_node.index.y == neighbor_node.index.y:
                        if Pixel.compare_cost(neighbor_node, open_node):
                            self.open_list[open_index] = neighbor_node
                        poplist = True
                
                for closed_index, closed_node in enumerate(self.closed_list):
                    if closed_node.index.x == neighbor_node.index.x and closed_node.index.y == neighbor_node.index.y:
                        if Pixel.compare_cost(neighbor_node, closed_node):
                            poplist = True
                
                if poplist==False and self.data[neighbor_node.index.x][neighbor_node.index.y]!=const.OBSTACLE:
                    self.open_list.append(neighbor_node)
                    if neighbor_node.index!=self.start and neighbor_node.index!=self.end:
                        self.data[neighbor_node.index.x][neighbor_node.index.y] = const.OPEN
                
                self.closed_list.append(curr_node)
                if curr_node!=self.start and curr_node!=self.end:
                    self.data[curr_node.index.x][curr_node.index.y] = const.CLOSED
                
                # plt.clf()
                # plt.imshow(self.data, interpolation = 'nearest')
                # plt.plot()
                # plt.pause(0.01)
        
                
                
    def get_path(self, last : 'Pixel'):
        temp = last
        path_points = []
        plt.ion()
        path_plot = plt.figure()
        while temp.parent!=None:
            self.data[temp.index.x][temp.index.y] = const.SOURCE
            # plt.clf()
            # plt.imshow(self.data, interpolation='nearest')
            # plt.plot()
            # plt.pause(0.00001)
            #print(last)
            path_points.append((temp.index.x, temp.index.y))
            temp = temp.parent
        # plt.imshow(self.data, interpolation = 'nearest')
        plt.savefig('result.png')
        return path_points
        #print(path_points)
        
    
    def return_protocol(self, path_points):
        protocol = ''
        for x in range(len(path_points)-1):
            if path_points[x][0] == path_points[x+1][0] - 1:
                protocol += ('W')
            elif path_points[x][0] == path_points[x+1][0] + 1:
                protocol += ('E')
            elif path_points[x][1] == path_points[x+1][1] - 1:
                protocol += ('S')
            elif path_points[x][1] == path_points[x+1][1] + 1:
                protocol += ('N')
        return protocol
        
        
        