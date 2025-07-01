# python -m venv .venv
import gymnasium as gym
import browsergym.core   # 必须导入，才会把环境注册到 Gym

def main() -> None:
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": "https://www.google.com"},
        headless=False,              # 设 True 可无头运行
        wait_for_user_message=False  # 不需要手动确认即可开始
    )

    try:
        obs, info = env.reset(seed=42)

        # BrowserGym 的动作是一串 **字符串**，
        # 语法来自官方动作原语（见下文表格）
        actions = [
            "click('btnK')",               # 点击 Google 搜索按钮
            "keyboard_type('hello world')"
        ]
        for act in actions:
            obs, reward, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                break
    finally:
        env.close()

if __name__ == "__main__":
    main()