import copy
import json
import numpy as np
from collections import deque, defaultdict
import pdb
from tasks.prompts import *

path = []
tree_list = []

def dfs(node):
    # 深度优先
    # if node.children == {}:
    if node is None:
        return
    for step, node in node.children.items():
        path.append(step)
        dfs(node)


def level_order_traversal(node):
    #层次遍历 存储treelist
    if node is None:
        return
    que = []
    que.append(node)
    level_path = []
    level_node = []
    tree_list = []
    while que:
        level_size = len(que)
        curlevel = defaultdict(dict)
        curlevel_node = []
        for _ in range(level_size):
            tnode = que.pop(0)
            # pdb.set_trace()
            # curlevel['step'] = tnode.y
            curlevel[tnode.history_path]['value'] = tnode.V
            curlevel_node.append(tnode)
            if tnode.parent is None:
                parent_id = None
            else:
                parent_id = tnode.parent.id
            node_dict = {
                'id': tnode.id,
                'depth': tnode.depth,
                'parent': parent_id,
                'children': tnode.children_list,
                'node_details':{"node_text":tnode.node_text, 'N':tnode.numVisits, "pre_relation": tnode.pre_relation, "step_value":tnode.V, "meta_info": \
                {'history_step_info': tnode.history_path, \
                    'final_ans_flag': tnode.final_ans_flag}},
                }
            tree_list.append(node_dict)
            for _, child in tnode.children.items():
                que.append(child)
        level_path.append(curlevel)
        level_node.append(curlevel_node)
    return level_path, level_node, tree_list


def count_dfs(node):
    if node.children == {}:
        return 1
    count = 1
    for _, node in node.children.items():
        count += count_dfs(node)
    return count

class treeNode(object):
    
    _id_counter = 0

    def __init__(self, node_text=None, parent=None, history_path='', pre_relation='', depth=0):
        treeNode._id_counter += 1
        self.id = treeNode._id_counter
        self.node_text = node_text  # str
        self.history_path = history_path  # str
        self.pre_relation = pre_relation
        self.parent = parent  # treeNode

        self.numVisits = 0  # int
        self.V = 0.0  # float
        self.children = {}  # dict{str:treeNode}
        self.children_list = []
        self.depth = depth  # int
        self.isFullyExpanded = False  # expanded
        self.visit_sequence = 0
        self.final_ans_flag = 0
        self.reflection = ''
        self.isTerminal = False  # value acceptable
        self.on_final_route = False
        self.min_steps_to_correct = 1024
        self.summary = ''
        self.he = 0  # hard estimation
        self.se = 0  # soft estimation
        self.path = []
        # self.node = []
        self.node_num = 1
        self.rollpath = []
        self.maxdepth = 1
        self.tree_list = []
        self.rollout_ans_flag = False
        self.type = ''

    @classmethod
    def reset_class_variable(cls):
        cls._id_counter = 0  # 清零类变量

    def __str__(self):
        s = ["numVisits: %d" % self.numVisits, f'V:{self.V}', "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))

    def append_children(self, new_node_text: str, pre_relation: str):
        node = treeNode(node_text=new_node_text, parent=self, pre_relation=pre_relation, depth=self.depth + 1)
        node.update_y_from_parent()
        self.children.update({new_node_text: node})
        self.children_list.append(node._id_counter)
        return self

    def count_node(self):
        root = self
        without_rollnode_num = count_dfs(root)
        self.node_num = without_rollnode_num + self.count_rollnode()


    def count_rollnode(self):
        root = self
        rollnode_num = 0
        _, level_node_list, _ = level_order_traversal(root)
        for item in level_node_list:
            for node in item:
                rollpath_len = len(node.rollpath)
                if rollpath_len != 0:
                    rollnode_num += rollpath_len
        return rollnode_num


    def update_y_from_parent(self):
        # pdb.set_trace()
        if self.parent is None:
            self.history_path = self.node_text
        else:
            self.history_path = self.parent.history_path + ' -> ' + self.pre_relation  + ' -> ' + self.node_text


    # def update_y_from_parent_with_trace_back(self, index, plan_change):
    #     # pdb.set_trace()
    #     if self.parent is None:
    #         self.y = self.pcd
    #         self.reflection_path = self.pcd
    #     else:
    #         self.y = self.parent.y + self.pcd
    #         if index == 0 and plan_change:
    #             self.reflection_path = self.parent.reflection_path + " Hold on, there seems to be an issue with this reasoning path. Let's try a different approach to the solution." + self.pcd
    #         elif index == 0 and not plan_change:
    #             self.reflection_path = self.parent.reflection_path + " Wait, I should think more carefully; this step doesn't seem right. Let's reconsider this step." + self.pcd
    #         else:
    #             self.reflection_path = self.parent.reflection_path + self.pcd


    def update_value(self, value):
        self.V = value


    def update_reflection(self, reflection):
        self.reflection = reflection


    def getBestV(self):  # Gets the subtree maximum value node
        if not self.isFullyExpanded:
            return self, self.V
        max_V = self.V
        max_node = self
        for child in self.children.values():
            subNode, subValue = child.getBestV()
            if subValue >= max_V:
                max_V = subValue
                max_node = subNode
        return max_node, max_V


    def trace_path(self):
        curnode = self
        level_path, level_node, tree_list = level_order_traversal(curnode)
        self.path = level_path[:]
        self.maxdepth = len(level_node)
        self.tree_list = tree_list[:] # save tree


    def trace_route(self):  # trace route from terminal node to root
        cur_node = self
        while cur_node is not None:
            cur_node.on_final_route = True
            cur_node = cur_node.parent


    # def get_new_value_samples(self):  # get value samples from search tree (start from terminal node)
    #     if self.depth == 0:
    #         return []
    #     step_value = 1.0 / self.depth
    #     new_samples = []
    #     cur_node = self.parent
    #     while cur_node is not None:
    #         for child in cur_node.children.values():
    #             if child.on_final_route:
    #                 child_value = step_value * child.depth
    #                 new_item = {'steps': child.y, 'value': child_value}
    #                 new_samples.append(new_item)
    #             else:
    #                 child_value = max(step_value * (cur_node.depth - 1), 0)
    #                 new_item = {'steps': child.y, 'value': child_value}
    #                 new_samples.append(new_item)
    #         cur_node = cur_node.parent
    #     return new_samples

    # def get_all_end_root_nodes_vm(self, end_gate):
    #     end_nodes = []
    #     if self.isFullyExpanded:
    #         for child in self.children.values():
    #             end_nodes.extend(child.get_all_end_root_nodes_vm(end_gate))
    #         return end_nodes
    #     else:
    #         if self.V >= end_gate or self.reflection == '<end>':
    #             return [self]
    #         else:
    #             return []

    # def get_all_end_root_nodes_prm(self):
    #     end_nodes = []
    #     if self.isFullyExpanded:
    #         for child in self.children.values():
    #             end_nodes.extend(child.get_all_end_root_nodes_prm())
    #         return end_nodes
    #     else:
    #         if self.reflection == '<end>':
    #             return [self]
    #         else:
    #             return []

    # def get_all_value_samples_vm(self):
    #     full_value_samples = []
    #     if self.depth == 0:
    #         self.V = 0
    #     else:
    #         if self.he == 0:
    #             r = -1
    #         else:
    #             r = 1
    #         self.V = max(0, (1 - self.parent.V) * r / self.min_steps_to_correct + self.parent.V)
    #         full_value_samples.append({'steps': self.y, 'value': self.V})
    #     if self.isFullyExpanded:
    #         for child in self.children.values():
    #             if child.min_steps_to_correct < 1024:
    #                 sub_samples = child.get_all_value_samples_vm()
    #                 full_value_samples.extend(sub_samples)
    #     return full_value_samples

    # def get_full_value_samples_vm(self, end_leaf_nodes):
    #     for leaf in end_leaf_nodes:
    #         if leaf.min_steps_to_correct > 1:
    #             continue
    #         else:
    #             leaf.he = 1
    #             cur_node = leaf.parent
    #             while cur_node is not None:
    #                 cur_node.min_steps_to_correct = min(
    #                     [n.min_steps_to_correct for n in cur_node.children.values()]) + 1
    #                 cur_node.he = 1
    #                 cur_node = cur_node.parent
    #     for leaf in end_leaf_nodes:
    #         if leaf.min_steps_to_correct > 1:
    #             cur_node = leaf.parent
    #             while cur_node is not None and cur_node.min_steps_to_correct == 1024:
    #                 cur_node = cur_node.parent
    #             if cur_node is None:
    #                 continue
    #             else:
    #                 m = cur_node.min_steps_to_correct
    #                 cur_node = leaf
    #                 while cur_node.min_steps_to_correct == 1024:
    #                     cur_node.min_steps_to_correct = m
    #                     cur_node = cur_node.parent
    #         else:
    #             continue
    #     value_samples = self.get_all_value_samples_vm()
    #     return value_samples

    # def get_all_value_samples_prm(self):
    #     full_value_samples = []
    #     if self.on_final_route:
    #         full_value_samples.append({'steps': self.y, 'value': self.he})
    #         if self.isFullyExpanded:
    #             for child in self.children.values():
    #                 if child.on_final_route:
    #                     sub_samples = child.get_all_value_samples_prm()
    #                     full_value_samples.extend(sub_samples)
    #         return full_value_samples
    #     else:
    #         return []

    # def get_full_value_samples_prm(self, end_leaf_nodes):
    #     for leaf in end_leaf_nodes:
    #         cur_node = leaf.parent
    #         while cur_node is not None:
    #             cur_node.on_final_route = True
    #             cur_node = cur_node.parent
    #     for leaf in end_leaf_nodes:
    #         cur_node = leaf.parent
    #         while cur_node is not None:
    #             cur_node.he = max([n.he for n in cur_node.children.values() if n.on_final_route])
    #             cur_node = cur_node.parent
    #     value_samples = self.get_all_value_samples_prm()
    #     return value_samples
