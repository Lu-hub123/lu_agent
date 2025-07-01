import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

@ray.remote(num_cpus=0.25)
class BrowserGymWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'BrowserGym')) #动态获取 webshop 文件夹的绝对路径
        sys.path.append(project_root)
        from .core import BrowserEnv  # noqa: WPS433 (runtime import)
        
        # env_kwargs['seed'] = seed
        self.env = gym.make("browsergym/openended") #**env_kwargs是一个字典参数展开，表示将用户传入的参数原样传给环境类的 __init__()。
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        # info = dict(info or {})  # make a *copy* so we can mutate safely 创建一个新的 dict 拷贝，防止修改原始的 info 对象
        # info['available_actions'] = self.env.get_available_actions() #是你手动添加的字段，值来自调用，创建的新的
        # info['task_score'] = reward

        # # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        # if done and reward == 1.0:
        #     info['won'] = True
        #     reward = 10.0
        # else:
        #     info['won'] = False
        #     reward = 0

        return obs, reward, terminated, truncated, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        # info = dict(info or {})
        # info['available_actions'] = self.env.get_available_actions()
        # info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render) #渲染的模式，返回的结果可能是图像，文本格式的状态
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()
