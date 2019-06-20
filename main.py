import argparse
from DQNAgent import DQNAgent
from GridWorld import GridWorld


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate')
PARSER.add_argument('-gamma', '--discount_factor', default=1.0, type=float, help='Discounted factor')
PARSER.add_argument('-e', '--epsilon', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-op', '--update_operation', default=100, type=int, help='Update operation of network')
PARSER.add_argument('-ep', '--episode', default=1000, type=int, help='Training episode iteration number')
PARSER.add_argument('-ms', '--memory_size', default=1000, type=int, help='Replay memory size')
PARSER.add_argument('-bs', '--batch_size', default=32, type=int, help='Batch size')
PARSER.add_argument('-r', '--render', default=True, type=bool, help='Render')
PARSER.add_argument('-st', '--start_epi', default=10, type=int, help='Render')
ARGS = PARSER.parse_args()
print (ARGS)

def main():
    succ_cnt, fail_cnt = 0, 0
    for episode in range(ARGS.episode):
        state = env.reset()
        while True:
            # DQN모델에서 현재 state를 인자로 받아 action을 선택
            action = agent.choose_action(state)

            # Action을 취하여 환경에서 next state, reward, done 반환
            next_state, reward, done = env.step(action)            
            
            # Reward가 0인 invalid한 action일때는 memory에 저장하지 않도록 구성.
            if reward is not 0:
                agent.add_memory(state, action, reward, next_state)

            # Next state를 state에 저장.
            state = next_state
            if done == 'DONE':
                succ_cnt += 1
                break
            elif done == 'FAIL':
                fail_cnt += 1
                break
            
        # Replay memory stack 후 진행해야 하며, episode가 10번 실행 후에 agent 학습
        if (episode >= ARGS.start_epi):
            agent.train()
            acc = succ_cnt/(succ_cnt + fail_cnt) * 100
            print ('episode: ', episode, ', acc: ', acc , '%')      
    env.grid_world.destroy()


if __name__ == "__main__":
    # GridWorld 
    env = GridWorld(
             render = ARGS.render
             )
    agent = DQNAgent(
             num_actions=env.num_actions, 
             num_features=env.num_features,
             update_operation=ARGS.update_operation,
             memory_size=ARGS.memory_size,
             batch_size=ARGS.batch_size,
             lr=ARGS.learning_rate,
             gamma=ARGS.discount_factor,
             epsilon=ARGS.epsilon
             )
    # give delay for GridWorld 
    env.grid_world.after(100, main)
    env.grid_world.mainloop()
