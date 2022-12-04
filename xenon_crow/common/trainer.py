import torch


class D3QNTrainer(object):
    def __init__(self, device) -> None:
        self.logger = self.__init_logger()
        self.device = torch.device(device)

    def __init_logger():
        return None

    def run(self, env, agent, max_episodes, max_steps):
        episode_rewards = []
        agent = agent.to(self.device)
        env = env.to(self.device)

        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0.0

            for step in range(max_steps):
                action, info = agent.get_action(state)
                next_state, reward, (done, trunc), info = env.step(action)
                agent.replay_buffer.push(
                    state, action, reward, next_state, done, info
                )
                episode_reward += reward

                if agent.replay_buffer.ready():
                    agent.update()

                if done or trunc:
                    episode_rewards.append(episode_reward)
                    print(f"[INFO]: Episode {episode}: {episode_reward}")
                    break

                state = next_state

        return episode_rewards
