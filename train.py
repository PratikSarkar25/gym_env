import gymnasium as gym
from stable_baselines3 import PPO

def main():
    env = gym.make("CarRacing-v3", continuous=True)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        batch_size=128,
        n_steps=1024,
        learning_rate=2.5e-4
    )

    try:
        model.learn(total_timesteps=100_000)
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # 🔥 FORCE SAVE (this WILL run)
    model.save("ppo_car_racing")
    print("Model saved as ppo_car_racing.zip")

    env.close()

if __name__ == "__main__":
    main()
