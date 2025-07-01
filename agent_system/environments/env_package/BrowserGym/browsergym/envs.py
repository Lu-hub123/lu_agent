import ray
import gymnasium as gym
import numpy as np
from typing import Optional
from pathlib import Path
from agent_system.environments.env_package.BrowserGym.browsergym.core.src.browsergym.core.action.highlevel import HighLevelActionSet
# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

@ray.remote(num_cpus=0.25)
class BrowserGymWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, action_mapping,use_raw_page_output,task_name,disable_env_checker,max_steps,headless,wait_for_user_message, extra_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'BrowserGym')) #动态获取 webshop 文件夹的绝对路径
        sys.path.append(project_root)
        from agent_system.environments.env_package.BrowserGym.browsergym.core.src.browsergym.core.env import BrowserEnv  # noqa: WPS433 (runtime import)
        
        # extra_kwargs['seed'] = seed 
        # env = gym.make("browsergym/openended") #**env_kwargs是一个字典参数展开，表示将用户传入的参数原样传给环境类的 __init__()。
        self.env = gym.make(  #这里的env应该已经是环境实例化的代名词了
            _get_env_name(task_name),
            disable_env_checker=disable_env_checker,
            max_episode_steps=max_steps,
            headless=headless,
            wait_for_user_message=wait_for_user_message,
            action_mapping=action_mapping,  # action mapping is provided by the agent
            use_raw_page_output=use_raw_page_output,
            **extra_kwargs,
        )
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely 创建一个新的 dict 拷贝，防止修改原始的 info 对象
        info['available_actions'] = "keyboard_type('hello world')" #是你手动添加的字段，值来自调用，创建的新的
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if terminated or truncated and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, terminated, truncated, info
    
    
    def reset(self, idx: int | None = None):
        # BrowserEnv.reset 返回 (obs, info)
        obs, info = self.env.reset(seed=idx)
        info = dict(info or {})            # 避免 NoneType

        # 如果需要暴露动作集合，取消下一行的注释
        # info["available_actions"] = self.env.get_available_actions()

        info["won"] = False
        return obs, info

    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    # def get_available_actions(self):
    #     """
    #     返回高层动作名称列表（字符串），便于上层 UI 做下拉菜单或按钮。
    #     """
    #     try:
    #         return HighLevelActionSet().action_names
    #     except AttributeError:
    #         # 兼容旧版 HighLevelActionSet
    #         return [fn for fn in dir(HighLevelActionSet) if not fn.startswith("_")]
    
    def get_goals(self):
        obs, _ = self.reset() #只想获得obs，不关心第二个返回值
        # BrowserGym 内部把目标保存在 self.goal_object；obs 里同样有
        return obs["goal_object"]
    
    # def initialize_with_goals(self, goals):
    #     self.task_kwargs["goal"] = goals
    #     return self.reset()

    def close(self):
        """Close the environment"""
        self.env.close()


class BrowserGymMultiProcessEnv(gym.Env):
    """Vectorised BrowserGym via **Ray** actors (similar to WebShopMultiProcessEnv)."""

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 5,
        is_train: bool = True,
        task_name: str = "openended",
        task_seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        headless: bool = False,
        record_video: bool = False,
        wait_for_user_message: bool = False,
        viewport: Optional[dict] = None,  # use default value from BrowserGym
        slow_mo: Optional[int] = None , # use default value from BrowserGym
        storage_state: Optional[str | Path | dict] = None,
        task_kwargs: Optional[dict] = None,  # use default value from BrowserGym
        action_mapping=HighLevelActionSet.to_python_code,
        use_raw_page_output = False,
        disable_env_checker=True,
        extra_kwargs = {}
        
        # env_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train

        self._rng = np.random.RandomState(seed)
        self.task_name = task_name
        self.task_seed = task_seed
        self.max_steps = max_steps
        self.headless = headless
        self.record_video = record_video
        self.wait_for_user_message = wait_for_user_message
        self.viewport = viewport  # use default value from BrowserGym if None
        self.slow_mo = slow_mo  # use default value from BrowserGym if None
        self.storage_state = storage_state
        self.task_kwargs = task_kwargs  # use default value from BrowserGym if None
        self.action_mapping = action_mapping
        self.use_raw_page_output = use_raw_page_output
        self.disable_env_checker = disable_env_checker
        # _env_kwargs = env_kwargs or {}    
        self.extra_kwargs = extra_kwargs
        

        # Spin up Ray actors
        self._workers = []

        for i in range(self.num_processes):
            worker = BrowserGymWorker.remote(seed + (i // self.group_n), self.action_mapping,self.use_raw_page_output,self.task_name,self.disable_env_checker,self.max_steps,self.headless,self.wait_for_user_message, self.extra_kwargs)
            self._workers.append(worker)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        print(goals_future)
        goals = ray.get(goals_future)
        print(goals)

        # # Initialize the remaining workers 
        # init_futures = []
        # for i in range(1, self.num_processes):
        #     init_futures.append(self._workers[i].initialize_with_goals.remote(goals))
        # ray.get(init_futures)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------


def _get_env_name(task_name: str):
    """Register tasks if needed (lazy import) and return environment name."""

    # lazy benchmark import
    if task_name.startswith("miniwob"):
        import browsergym.miniwob     #todo
    elif task_name.startswith("workarena"):
        import browsergym.workarena
    elif task_name.startswith("webarena"):
        import browsergym.webarena
    elif task_name.startswith("visualwebarena"):
        import browsergym.visualwebarena
    elif task_name.startswith("assistantbench"):
        import browsergym.assistantbench
    elif task_name.startswith("weblinx"):
        import weblinx_browsergym

    return f"browsergym/{task_name}"



def build_browsergym_envs(
    task_name: str,
    task_seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    headless: bool = False,
    record_video: bool = False,
    wait_for_user_message: bool = False,
    viewport: Optional[dict] = None,  # use default value from BrowserGym
    slow_mo: Optional[int] = None , # use default value from BrowserGym
    storage_state: Optional[str | Path | dict] = None,
    task_kwargs: Optional[dict] = None,  # use default value from BrowserGym
    action_mapping=HighLevelActionSet.to_python_code,
    use_raw_page_output = False,
    # exp_dir=self.exp_dir
):
    # for openended task, set environment and agent to interactive chat mode on a start url
    if task_name == "openended":
        # agent_args.chat_mode = True
        # env_args.wait_for_user_message = True
        task_kwargs = {"start_url": "https://www.google.com"}
    # record_video: bool = False,
    # wait_for_user_message: bool = False,
    # viewport: Optional[dict] = None,  # use default value from BrowserGym
    # slow_mo: Optional[int] = None , # use default value from BrowserGym
    # storage_state: Optional[str | Path | dict] = None,
    # task_kwargs: Optional[dict] = None,  # use default value from BrowserGym
    # action_mapping=HighLevelActionSet.to_python_code,
    # use_raw_page_output = False,
    # exp_dir=self.exp_dir
    extra_kwargs = {}
    # if record_video:
    #     extra_kwargs["record_video_dir"] = exp_dir
    if viewport:
        extra_kwargs["viewport"] = viewport
    if slow_mo is not None:
        extra_kwargs["slow_mo"] = slow_mo
    if storage_state:
        extra_kwargs["pw_context_kwargs"] = {"storage_state": storage_state}
    if task_kwargs is not None:
        extra_kwargs["task_kwargs"] = task_kwargs
    # if exp_task_kwargs:
    #     extra_kwargs["task_kwargs"] = extra_kwargs.get("task_kwargs", {}) | exp_task_kwargs

        # assistantbench hack, write the task output (agent prediction) to a file in the experiment's directory
        # TODO: find a better way to deal with this
        # if task_name.startswith("assistantbench.test"):
        #     extra_kwargs["task_kwargs"] = extra_kwargs.get("task_kwargs", {}) | {
        #         "output_file": exp_dir / "assistantbench-prediction.json"
        #     }
    # print("extra_kwargs:", extra_kwargs)
    

    return BrowserGymMultiProcessEnv(
        action_mapping=action_mapping,
        use_raw_page_output=use_raw_page_output,
        task_name=task_name,   # ← 单独传
        disable_env_checker=True,
        max_steps=max_steps,
        headless=headless,
        wait_for_user_message=wait_for_user_message,
        extra_kwargs=extra_kwargs,            # ← 如果你真想把它整个传进来
)

