from torch import tensor, float32, is_tensor
from tqdm.auto import trange


class D3QNTrainer(object):
    def __init__(self) -> None:
        self.logger = self.__init_logger()

    def __init_logger(self):
        return None

    def run(self, env, agent, max_episodes, update_inter):
        episode_rewards = []
        progress_bar = trange(
            max_episodes, ncols=150, desc="Training", position=0, leave=True
        )
        for ep in progress_bar:
            step, reward, total_loss = 1, 0.0, 0.0
            terminated = False
            state, _ = env.reset()

            while not terminated:
                if not is_tensor(state):
                    state = tensor(state, dtype=float32).unsqueeze(0)

                action = agent.get_action(state.float())
                next_state, r, done, trunc, *_ = env.step(action)
                terminated = done or trunc

                agent.replay_buffer.push((state, action, r, next_state, terminated))
                reward += r

                if agent.replay_buffer.ready() and step % update_inter == 0:
                    total_loss += agent.update()

                step += 1
                state = next_state
                progress_bar.set_description(f"Training {step}/1000", refresh=True)

            
            agent.update_epsilon(agent.epsilon * 0.997)
            total_loss /= step
            episode_rewards.append(reward)
            progress_bar.set_postfix(reward=reward, epsilon=agent.epsilon, loss=total_loss, refresh=True)

            if ep == max_episodes-15:
                env.save_masks = True

        return episode_rewards


class ReinforceTrainer(object):
    def __init__(self) -> None:
        self.logger = self.__init_logger()

    def __init_logger(self):
        return None

    @classmethod
    def generate_episode(cls, env, agent):
        step, reward = 1, 0.0
        terminated = False
        state, _ = env.reset()

        while not terminated:
            if not is_tensor(state):
                state = tensor(state, dtype=float32).unsqueeze(0)

            action, log_prob, value = agent.get_action(state.float())
            next_state, r, done, trunc, *_ = env.step(action)
            terminated = done or trunc

            reward += r
            agent.replay_buffer.push((log_prob, value, r))

            step += 1
            state = next_state

        return reward

    def run(self, env, agent, max_episodes):
        episode_rewards = []
        progress_bar = trange(
            max_episodes, ncols=150, desc="Training", position=0, leave=True
        )
        ma_reward = -10
        for _ in progress_bar:
            reward = ReinforceTrainer.generate_episode(env, agent)
            episode_rewards.append(reward)
            loss = agent.update()
            ma_reward = 0.05 * reward + (1 - 0.05) * ma_reward
            progress_bar.set_postfix(ma_reward=ma_reward, reward=reward, loss=loss, refresh=True)

            if ma_reward > env.spec.reward_threshold:
                print("Solved! Running reward: {}".format(ma_reward))
                break

        return episode_rewards


class A2CTrainer(object):
    def __init__(self) -> None:
        self.logger = self.__init_logger()

    def __init_logger(self):
        return None

    @classmethod
    def generate_episode(cls, env, agent):
        step, reward, ep_entropy = 1, 0.0, 0.0
        terminated = False
        state, _ = env.reset()

        while not terminated:
            if not is_tensor(state):
                state = tensor(state, dtype=float32).unsqueeze(0)

            action, log_prob, entropy, value = agent.get_action(state.float())
            next_state, r, done, trunc, *_ = env.step(action)
            terminated = done or trunc

            agent.replay_buffer.push((log_prob, value, r))
            reward += r
            ep_entropy += entropy

            step += 1
            state = next_state

        return reward, ep_entropy

    def run(self, env, agent, max_episodes):
        ma_reward, episode_rewards = -100.0, []
        progress_bar = trange(
            max_episodes, ncols=150, desc="Training", position=0, leave=True
        )
        for _ in progress_bar:
            reward, entropy = A2CTrainer.generate_episode(env, agent)
            ma_reward = 0.05 * reward + (1 - 0.05) * ma_reward
            episode_rewards.append(reward)
            loss = agent.update(entropy)

            progress_bar.set_postfix(reward=reward, loss=loss, ma_reward=ma_reward, refresh=True)
            if ma_reward > env.spec.reward_threshold:
                print("Solved! Running reward: {}".format(ma_reward))
                break

        return episode_rewards
