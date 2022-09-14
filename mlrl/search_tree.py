from abc import ABC, abstractmethod
from typing import Callable, Any, List, Tuple, Dict
from collections import defaultdict

import gym
import numpy as np


class ObjectState(ABC):
    
    @staticmethod
    @abstractmethod
    def extract_state(env: gym.Env) -> 'ObjectState':
        pass
    
    @abstractmethod
    def set_environment_to_state(self, env: gym.Env):
        pass
    
    @abstractmethod
    def get_state_vector(self) -> np.array:
        pass


class SearchTreeNode:
    __slots__ = ['parent', 'node_id', 'state', 'action', 'reward', 'children', 'done']
    
    def __init__(self, 
                 node_id: int,
                 parent: 'SearchTreeNode', 
                 state: ObjectState, 
                 action: int, 
                 reward: float, 
                 done: bool):
        self.node_id = node_id
        self.parent = parent
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.children: Dict[int, List['SearchTreeNode']] = defaultdict(list)
        
    def expand_node(self, env: gym.Env, action: int, new_node_id: int) -> 'SearchTreeNode':
        if self.done:
            raise Exception('Cannot expand a terminal node')

        self.state.set_environment_to_state(env) 
            
        obs, reward, done, *_ = env.step(action)
        next_state: ObjectState = self.state.extract_state(env)
        
        child_node = SearchTreeNode(new_node_id, self, next_state, action, reward, done)
        self.children[action].append(child_node)
        return child_node

    def get_id(self) -> int:
        return self.node_id

    def get_parent(self) -> 'SearchTreeNode':
        return self.parent

    def get_parent_id(self) -> int:
        if self.parent:
            return self.parent.node_id
        return -1

    def get_children(self) -> Dict[int, List['SearchTreeNode']]:
        return self.children

    def can_expand(self) -> bool:
        return not self.done
    
    def get_trajectory(self) -> Tuple[int, float, ObjectState]:
        return (self.action, self.reward, self.state)
    
    def get_state(self) -> ObjectState:
        return self.state

    def get_action(self) -> int:
        return self.action or -1
    
    def get_preceding_action(self) -> int:
        return self.action
    
    def get_reward_received(self) -> float:
        return self.reward

    def __repr__(self) -> str:
        return f'action={self.action}, reward={self.reward}, state={self.state}, children={self.children}'


class SearchTree:
    
    def __init__(self, env: gym.Env, extract_state: Callable):
        self.env = env
        
        self.root_node: SearchTreeNode = SearchTreeNode(0, None, extract_state(env), None, 0, False)
        self.node_list: List[SearchTreeNode] = [self.root_node]
    
    def expand(self, node_idx: int, action: int):
        node = self.node_list[node_idx]
        if node.can_expand():
            child_node = node.expand_node(self.env, action, len(self.node_list))
            self.node_list.append(child_node)
            return True
        return False
        
    def get_nodes(self) -> List[SearchTreeNode]:
        return self.node_list

    def get_num_nodes(self) -> int:
        return len(self.node_list)
    
    def get_root(self) -> SearchTreeNode:
        return self.root_node
    
    def __repr__(self) -> str:
        return str(self.root_node)
