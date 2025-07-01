# # test_browsergym_on_verl.py
# import ray
# import numpy as np

# # ① —— veRL-agent 封装层

# from agent_system.environments.env_package.BrowserGym.browsergym.envs import build_browsergym_envs
# from agent_system.environments.env_package.BrowserGym.browsergym.projection import browsergym_projection

# from agent_system.environments.env_manager import (
#     BrowserGymEnvironmentManager,   # 统一接口管理器
# )

# # ② —— 给底层 BrowserEnv 的 kwargs（按需改动即可）
# env_kwargs = {
#     "task_entrypoint": "browsergym.tasks.WebQA",   # 官方自带问答任务
#     "headless": True,                              # False 会弹出浏览器窗口
#     "viewport": {"width": 1280, "height": 720},
#     "timeout": 10_000,                             # playwright 超时 (ms)
# }

# # ③ —— 构造 1 个环境（无并行 / 无 rollout 组）并包一层 Manager
# vector_env = build_browsergym_envs(
#     seed=42,
#     env_num=1,
#     group_n=1,
#     is_train=True,
#     env_kwargs=env_kwargs,
# )
# env_mgr = BrowserGymEnvironmentManager(vector_env, browsergym_projection, "browsergym/WebQA")

# # ④ —— reset → step → 简单打印
# obs, infos = env_mgr.reset()
# print("▸ prompt preview:\n", obs["text"][0][:300], "…\n")

# # 随便给一个 no-op 动作：<think>...</think><action>wait[1]</action>
# dummy = ["<think>just wait</think><action>wait[1]</action>"]
# obs, rewards, dones, infos = env_mgr.step(dummy)

# print("reward:", rewards, "done:", dones)
# env_mgr.close()
# ray.shutdown()

# browsergym experiments utils
from agent_system.environments.env_package.BrowserGym.browsergym.experiments.src.browsergym.experiments.loop import EnvArgs, ExpArgs, get_exp_result
import argparse
# locally defined agent
from agent_system.environments.env_package.BrowserGym.demo_agent.agent import DemoAgentArgs
from agent_system.environments.env_package.BrowserGym.browsergym.core.src.browsergym.core.env import BrowserEnv
from agent_system.environments.env_package.BrowserGym.browsergym.core.src.browsergym.core.task import OpenEndedTask

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="abc",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--visual_effects",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html",
        type=str2bool,
        default=False,
        help="Use HTML in the agent's observation space.",
    )
    parser.add_argument(
        "--use_axtree",
        type=str2bool,
        default=True,
        help="Use AXTree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in the agent's observation space.",
    )

    return parser.parse_args()
def main():
    print(
        """\
--- WARNING ---
This is a basic agent for demo purposes.
Visit AgentLab for more capable agents with advanced features.
https://github.com/ServiceNow/AgentLab"""
    )
    args = parse_args()
    # setting up agent config
    agent_args = DemoAgentArgs(
        model_name=args.model_name, #TODO
        chat_mode=False,
        demo_mode="default" if args.visual_effects else "off",
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=args.use_screenshot,
    )

    # setting up environment config
    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=3,
        headless=False,  # keep the browser open
        # viewport={"width": 1500, "height": 1280},  # can be played with if needed
    ) #这几步都是在传入参数

    # for openended task, set environment and agent to interactive chat mode on a start url
    if args.task_name == "openended":
        # agent_args.chat_mode = True
        # env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}
        

    # setting up the experiment
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    ) #要找在哪里定义action要做什么？？找到了highlevel的代码处
    # 这一步也是在传入参数

    # running and logging results
    exp_args.prepare("./results") #这一步才开始行动 运行
    exp_args.run() #这一步才开始行动
    # from agent_system.environments.env_package.BrowserGym.browsergym.experiments.src.browsergym.experiments.loop import ExpArgs #这个是应该import类还是py文件？？？
    # print(exp_args.action)
    # action = "scroll"
    # a = BrowserEnv(task_entrypoint=OpenEndedTask(start_url = args.start_url,seed = None))
    # obs = a.reset()
    # reward,obs  = a.step(action)  #直接写死action就可以跳过agent直接执行动作
    # print("Reward:", reward)
    # print("Terminated?", obs)
    # loading and printing results
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()