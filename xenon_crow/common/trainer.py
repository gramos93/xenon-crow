

class Trainer(object):
    def __init__(self) -> None:
        self.logger = self.__init_logger()

    def __init_logger():
        return None

    def run(self, env, agent, max_episodes, max_steps, batch_size):
        episode_rewards = []

        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward

                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)   

                if done or step == max_steps-1:
                    episode_rewards.append(episode_reward)
                    print("Episode " + str(episode) + ": " + str(episode_reward))
                    break

                state = next_state

        return episode_rewards