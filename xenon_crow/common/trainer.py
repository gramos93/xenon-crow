from torch import tensor, float32
from tqdm.auto import trange

class D3QNTrainer(object):
    def __init__(self) -> None:
        self.logger = self.__init_logger()

    def __init_logger(self):
        return None

    def run(env, agent, max_episodes, update_inter):
        episode_rewards = []
        progress_bar = trange(
            max_episodes,
            ncols=150,
            desc="Training",
            position=0,
            leave=True
        )
        for _ in progress_bar:
            step, reward = 1, 0.0
            terminated = False
            state, _ = env.reset()

            while not terminated:
                action = agent.get_action(tensor(state, dtype=float32).unsqueeze(0))
                next_state, r, done, trunc, *_ = env.step(action)
                terminated = done or trunc

                agent.replay_buffer.push(
                    (state, action, r, next_state, terminated)
                )
                reward += r

                if agent.replay_buffer.ready() and step % update_inter == 0 :
                    agent.update()

                step += 1
                state = next_state
            
            agent.update_epsilon(agent.epsilon*0.99)
            episode_rewards.append(reward)
            progress_bar.set_postfix(reward=reward, epsilon=agent.epsilon, refresh=True)

        return episode_rewards
