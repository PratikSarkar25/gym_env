import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # Create environment WITH rendering
    env = gym.make(
        "CarRacing-v3",
        render_mode="human",   # THIS is what opens the window
        continuous=True
    )

    # Load trained model
    model = PPO.load("ppo_car_racing")

    obs, info = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()

if __name__ == "__main__":
    main()
