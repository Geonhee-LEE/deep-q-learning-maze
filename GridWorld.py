import numpy as np
import tkinter 
import time

X = 0   # X 좌표
Y = 1   # Y 좌표

UP = 0      # Action UP
DOWN = 1    # Action DOWN
LEFT = 2    # Action LEFT
RIGHT = 3   # Action RIGHT

class GridWorld(object):
    # Grid world 초기화.
    def __init__(self, render):
        self._render = render
        self.total_reward = 0
        self.agent_pos = np.array([0, 0])
        self.state = (0, 0, 'start')
        self.height = 3
        self.width = 4
        self.action = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.action)
        self.num_features = 2
        self.wall_pose = np.array([1, 1])
        self.neg_reward_pose = np.array([3, 1])
        self.pos_reward_pose = np.array([3, 2])
        self.build_gridworld()

    # TKinter를 이용하여 GridWorld 구축
    def build_gridworld(self):
        # TKinter instance 생성
        self.grid_size = 120
        self.grid_world=tkinter.Tk()
        self.grid_world.geometry('{0}x{1}'.format(self.width * self.grid_size, self.height * self.grid_size))        
        self.grid_world.title("Geonhee's GridWorld")
        self.canvas = tkinter.Canvas(self.grid_world,  bg='white', height=self.height * self.grid_size, width=self.width * self.grid_size)

        # Agent 시작 grid (0, 0) 
        grid_start_pos = np.array([self.grid_size*0.5, self.grid_size*0.5])
        self.agent = self.canvas.create_oval(
            grid_start_pos[X] - grid_start_pos[X]*0.3, grid_start_pos[Y] - grid_start_pos[Y]*0.3,
            grid_start_pos[X] + grid_start_pos[X]*0.3, grid_start_pos[Y] + grid_start_pos[Y]*0.3,
            fill='blue')

        # Wall grid (wall_pose)         
        wall_pos = grid_start_pos + self.grid_size * self.wall_pose
        self.wall_pos = self.canvas.create_rectangle(
            wall_pos[X] - self.grid_size*0.5,  wall_pos[Y] - self.grid_size*0.5, 
            wall_pos[X] + self.grid_size*0.5, wall_pos[Y] + self.grid_size*0.5,
            fill='black')

        # Negative reward terminal (neg_reward_pose)
        neg_terminal = grid_start_pos + self.grid_size * self.neg_reward_pose
        self.neg_terminal = self.canvas.create_rectangle(
            neg_terminal[X] - self.grid_size*0.5, neg_terminal[Y] - self.grid_size*0.5,
            neg_terminal[X] + self.grid_size*0.5, neg_terminal[Y] + self.grid_size*0.5,
            fill='red')
        self.canvas.create_text(self.grid_size * self.neg_reward_pose[X] +self.grid_size*0.5 , self.grid_size * self.neg_reward_pose[Y] +self.grid_size*0.5,fill="black",font="Times 36 bold",
                        text="-1")

        # Positive reward terminal (pos_reward_pose)
        pos_terminal = grid_start_pos + self.grid_size * self.pos_reward_pose
        self.pos_terminal = self.canvas.create_rectangle(
            pos_terminal[X] - self.grid_size*0.5, pos_terminal[Y] - self.grid_size*0.5,
            pos_terminal[X] + self.grid_size*0.5, pos_terminal[Y] + self.grid_size*0.5,
            fill='yellow')
        self.canvas.create_text(self.grid_size * self.pos_reward_pose[X] +self.grid_size*0.5, self.grid_size * self.pos_reward_pose[Y] +self.grid_size*0.5,fill="black",font="Times 36 bold",
                        text="+1")

        # 경계 line 생성
        for col in range(0, self.width * self.grid_size, self.grid_size): # Grid 수평선 생성
            ver_st_x, ver_st_y, ver_des_x, ver_des_y = col, 0, col, self.height * self.grid_size 
            self.canvas.create_line(ver_st_x, ver_st_y, ver_des_x, ver_des_y)
        for row in range(0, self.height * self.grid_size, self.grid_size):  # Grid 수직선 생성
            ver_st_x, ver_st_y, ver_des_x, ver_des_y = 0, row, self.width * self.grid_size, row
            self.canvas.create_line(ver_st_x, ver_st_y, ver_des_x, ver_des_y)

        self.canvas.pack()
        self.grid_world.update()

    # GridWorld 초기화
    def reset(self):
        self.canvas.delete(self.agent)
        grid_start_pos = np.array([self.grid_size*0.5, self.grid_size*0.5])
        self.agent = self.canvas.create_oval(
            grid_start_pos[X] - grid_start_pos[X]*0.3, grid_start_pos[Y] - grid_start_pos[Y]*0.3,
            grid_start_pos[X] + grid_start_pos[X]*0.3, grid_start_pos[Y] + grid_start_pos[Y]*0.3,
            fill='blue')
        self.total_reward = 0
        self.state = (0, 0, 'start')
        self.agent_pos = self.extract_pos(self.state)
        return self.agent_pos

    # Action을 인자로 받아 next state, reward, done 반환
    def step(self, action):
        next_state = self.update_state(action)       # action을 취했을때 이 aciton이 유효한지 확인하고, 유효하다면 action에 따른 state update
        reward = self.get_reward()
        done = self.status()              
        next_grid_state = self.extract_pos(next_state)
        self.total_reward += reward
        return next_grid_state, reward, done

    # Next STATE에서의 status return
    def status(self):
        agent_x, agent_y, _ = self.state
        if agent_x == self.pos_reward_pose[X] and agent_y == self.pos_reward_pose[Y]:
            return 'DONE'   
        elif agent_x == self.neg_reward_pose[X] and agent_y == self.neg_reward_pose[Y]:
            return 'FAIL'         
        return 'NOT DONE'

    # Action을 통해 도달하는 state에서 획득하는 reward return
    def get_reward(self):
        agent_x, agent_y, mode = self.state
        if agent_x == self.pos_reward_pose[X] and agent_y == self.pos_reward_pose[Y]:
            return 1.0
        elif agent_x == self.neg_reward_pose[X] and agent_y == self.neg_reward_pose[Y]:
            return -1.0

        # 1-step 이동시 페널티 제공
        if mode == 'valid':
            return -0.04    
        else:
            return 0

    # Action을 인자로 받아 유효한지 확인하고, 유효하다면 action에 따른 state update
    def update_state(self, action):
        x, y, nmode = self.state
        valid_actions = self.valid_actions()    # 유효한 action만 추출.   

        if not valid_actions:                   # 유효한 action이 없을 때
            nmode = 'blocked'
        elif action in valid_actions:           # 유효한 action 중에서 선택한 action으로 state update
            nmode = 'valid'
            if action == LEFT:
                x -= 1
            elif action == UP:
                y -= 1
            if action == RIGHT:
                x += 1
            elif action == DOWN:
                y += 1
        else:                                   # 유효한 action중에서 선택한 action이 없을 때.
            nmode = 'invalid'
    
        if self._render == True and nmode == 'valid':   # render option이 True인 경우에 grid world를 그리기 위해서 agent 위치 값 갱신
            self.move_agent(self.state, action)
                     
        self.state = (x, y, nmode)  # Next state   

        return self.state
        

    def move_agent(self, state, action):
        # 선택한 action에 대해서 agent가 grid world에서 이동 .
        base_action = np.array([0, 0])
        if action == UP: #up
            base_action[Y] -= self.grid_size
        elif action == DOWN: #down
            base_action[Y] += self.grid_size
        elif action == LEFT: #left
            base_action[X] -= self.grid_size
        elif action == RIGHT: #right
            base_action[X] += self.grid_size
        self.canvas.move(self.agent, base_action[X], base_action[Y])
        
        if self._render == True:
            self.render()
        else :
            self.grid_world.destroy()

    def extract_pos(self, state):
        # 현재 state를 grid world로 표현하기 위해 변환.
        self.agent_pos[X] = state[X]
        self.agent_pos[Y] = state[Y]        
        return self.agent_pos

    def valid_actions(self):
        x, y, _ = self.state # 현재 agent state 위치 
        actions = [UP, DOWN, LEFT, RIGHT] 

        # 제일 위쪽에 있을 때, 'UP' action 제거
        if y == 0:
            actions.remove(UP)
        # 제일 아래쪽에 있을 때, 'RIGHT' action 제거            
        elif y == self.height-1:
            actions.remove(DOWN)
        # 제일 왼쪽에 있을 때, 'LEFT' action 제거
        if x == 0:
            actions.remove(LEFT)
        # 제일 오른쪽에 있을 때, 'RIGHT' action 제거
        elif x == self.width-1:
            actions.remove(RIGHT)

        # Wall이 위쪽에 있을 때, 'UP' action 제거
        if y >= 0 and x == self.wall_pose[X] and y-1 == self.wall_pose[Y]:
            actions.remove(UP)
        # Wall이 아래쪽에 있을 때, 'DOWN' action 제거
        if y < self.height-1 and x == self.wall_pose[X] and y+1 == self.wall_pose[Y]:
            actions.remove(DOWN)
        # Wall이 왼쪽에 있을 때, 'LEFT' action 제거
        if x >= 0 and  x-1 == self.wall_pose[X] and y == self.wall_pose[Y]:
            actions.remove(LEFT)
        # Wall이 오른쪽에 있을 때, 'RIGHT' action 제거
        if x < self.width-1 and x+1 == self.wall_pose[X] and y == self.wall_pose[Y]:
            actions.remove(RIGHT)

        return actions        

    def render(self):
        time.sleep(0.01)
        self.grid_world.update()
