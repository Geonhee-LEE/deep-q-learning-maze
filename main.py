import argparse
from collections import deque
import numpy as np
from DQNAgent import DQNAgent
from GridWorld import GridWorld

# Hyperparameter 값 받을 수 있도록 구성.
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate')
PARSER.add_argument('-discount_factor', '--discount_factor', default=1.0, type=float, help='Discounted factor')
PARSER.add_argument('-e', '--epsilon', default=1.0, type=float, help='discount factor')
PARSER.add_argument('-ep', '--episode', default=1500, type=int, help='Training episode iteration number')
PARSER.add_argument('-ms', '--memory_size', default=5000, type=int, help='Replay memory size')
PARSER.add_argument('-bs', '--batch_size', default=32, type=int, help='Batch size')
PARSER.add_argument('-r', '--render', default=True, type=int, help='Render')
PARSER.add_argument('-mes', '--max_episode_steps', default=100, type=bool, help='Maximum step num')
ARGS = PARSER.parse_args()
print (ARGS)

avg_return_list = deque(maxlen=10)
avg_loss_list = deque(maxlen=10)

def main():
    for i in range(ARGS.episode):
        obs = env.reset()        
        done = ''
        total_reward = 0
        total_loss = 0
        for _ in range(ARGS.max_episode_steps):
            # DQN모델에서 현재 state를 인자로 받아 action을 선택
            action = agent.get_action(obs)

            # Action을 취하여 환경에서 next state, reward, done 반환
            next_obs, reward, done = env.step(action)

            # Replay memory에 experience 추가.
            if reward is not 0 :
                agent.add_experience(obs,action,reward,next_obs,done)
            
            # Prediction network 학습 및 policy 갱신(입실론값 변경)
            loss = agent.train()
            agent.update_policy()
                    
            obs = next_obs
            total_reward += reward
            total_loss += loss

            if done == 'DONE' or done == 'FAIL':
                break
                
        # Target network parameter 갱신
        agent.update_target()
        avg_return_list.append(total_reward)
        avg_loss_list.append(total_loss)      
        
        # Logging
        if (np.mean(avg_return_list) > 0.5): 
            print('[{}/{}] loss : {:.3f}, return : {:.3f}, eps : {:.3f}'.format(i,ARGS.episode, np.mean(avg_loss_list), np.mean(avg_return_list), agent.epsilon))
            print('The problem is solved with {} episodes'.format(i))
            break        
        if (i%100)==0:
            print('[{}/{}] loss : {:.3f}, return : {:.3f}, eps : {:.3f}'.format(i,ARGS.episode, np.mean(avg_loss_list), np.mean(avg_return_list), agent.epsilon))                  
    env.grid_world.destroy()


if __name__ == "__main__":
    # GridWorld class instance 생성
    env = GridWorld(
             render = ARGS.render
             )
    # DQN Agent class instance 생성
    agent = DQNAgent(
             n_action=env.n_action, 
             obs_dim=env.obs_dim,
             memory_size=ARGS.memory_size,
             batch_size=ARGS.batch_size,
             lr=ARGS.learning_rate,
             discount_factor=ARGS.discount_factor,
             epsilon=ARGS.epsilon
             )
    env.grid_world.after(100, main)
    env.grid_world.mainloop()
