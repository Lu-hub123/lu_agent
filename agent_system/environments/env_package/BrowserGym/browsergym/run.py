# test_openended.py
from agent_system.environments.env_package.BrowserGym.browsergym.core.src.browsergym.core.task import OpenEndedTask
from agent_system.environments.env_package.BrowserGym.browsergym.core.src.browsergym.core.env import BrowserEnv

task = OpenEndedTask(
    seed=0,
    start_url="https://www.wikipedia.org",
    goal="Exit once page is loaded.",
)

env = BrowserEnv(task_entrypoint=task.__class__, headless=True)
obs, _ = env.reset()          # 启动浏览器并打开 wiki
print("Got obs keys:", obs.keys())
env.step("<action>wait[1]</action>")   # 随便执行一步
