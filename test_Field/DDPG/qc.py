import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import NN_Models
import vis3D

device = NN_Models.device

pi,_ = NN_Models.gen_Deterministic_Actor(12,4,2)

class quard_Copter(object):
    
    def __init__(self):
        self.Speed = np.zeros(3)
        self.Acc = np.zeros(3)
        self.Angle = np.zeros(3)
        self.Force = np.zeros(4)
        self.pqr = np.zeros(3)
        self.dpqr = np.zeros(3)
        self.I = np.array([3.828,
                           3.828,
                           7.135])
        self.P = np.zeros(3)
        self.Mass = 0.4
        self.L = 0.205
        self.action_dim = 4
        self.state_dim = self.Speed.shape[0]+\
                         self.Angle.shape[0]+\
                         self.P.shape[0]+\
                         self.pqr.shape[0]
        self.action_lim = 2.
        self.dt = 0.05
        
    def set_speed(self,speed=np.random.rand(3)):
        self.Speed = speed   
#    def set_acceleration(self,acc=np.random.rand(3)):
#        self.Acc = acc     
    def set_location(self,loc=np.zeros(3)):
        self.P = loc
    def set_angle(self,angle=np.random.uniform(-0.2,0.2,3)):
        self.Angle = angle
    def set_pqr(self,pqr=np.random.randn(3)):
        self.pqr = pqr
    def set_I(self,I):
        self.I = I
#    def set_dpqr(self,dpqr=np.zeros(3)):
#        self.dpqr = dpqr
    def set_Force(self,force=2*np.random.rand(4)):
        self.Force = force
    def set_Mass(self,mass):
        self.Mass = mass
    def set_L(self,L):
        self.L = L

    def check_stable_stat(self,state,dist=np.zeros(3)):
        P,Speed,Angle,pqr = state
        
        
        phi_threshold = 0.5
        theta_threshold = 0.5
        safe_zone = np.array([2,2,2])


        reward = -np.sum((P-dist)**2)/3.-4*(Angle[0]-phi_threshold)**2-4*(Angle[1]-theta_threshold)**2
        done = False
        if np.abs(Angle[0]) > phi_threshold or\
           np.abs(Angle[1]) > theta_threshold or\
           np.abs(P[0]) > safe_zone[0] or \
           np.abs(P[1]) > safe_zone[1] or \
           np.abs(P[2]) > safe_zone[2]:
               done = True
               reward = -6.
        return reward,done

    def grad(self,Force,state):

        P,Speed,Angle,pqr = state
        
        sins = np.sin(Angle)
        coss = np.cos(Angle)
        
        mass = self.Mass
        
        Acc = np.zeros(3)
        Acc[0] = np.sum(Force*(sins[0]*sins[2]+coss[0]*sins[1]*coss[2]))/mass
        Acc[1] = np.sum(Force*(-sins[0]*coss[2]+coss[0]*sins[1]*sins[2]))/mass
        Acc[2] = np.sum(Force*(coss[0]*coss[1]))/mass-9.8
        
        p,q,r = pqr
        
        dAngle = np.zeros(3)
        dAngle[0] = (p*coss[1]+q*sins[0]*sins[1]+r*coss[0]*sins[1])/coss[1]
        dAngle[1] = q*coss[0]+r*sins[0]
        dAngle[2] = (q*sins[0]+r*coss[0])/coss[1]
        
        Ix,Iy,Iz = self.I
        L = self.L
        
        dpqr = np.zeros(3)
        dpqr[0] = (L*(Force[3]-Force[1])+q*r*(Iy-Iz))/Ix
        dpqr[1] = (L*(Force[2]-Force[0])+p*r*(Iz-Ix))/Iy
        dpqr[2] = (L*(Force[1]+Force[3]-Force[0]-Force[2])+q*r*(Ix-Iy))/Iz
        
        grad = [Speed,Acc,dAngle,dpqr]
        return grad

    def step(self,Force,reward_fn=check_stable_stat):
        self.Force = Force
        
        state_1 = [self.P,self.Speed,self.Angle,self.pqr]
        h = self.dt/4.
        k1 = self.grad(Force,state_1)
        state_2 = [state_1[i]+h*k1[i] for i in range(len(k1))]
        k2 = self.grad(Force,state_2)
        state_3 = [state_2[i]+h*k2[i] for i in range(len(k1))]
        k3 = self.grad(Force,state_3)
        state_4 = [state_3[i]+h*k3[i] for i in range(len(k1))]
        k4 = self.grad(Force,state_4)
        state = [state_1[i]+h/6.*(k1[i]+2*k2[i]+2*k3[i]+k4[i]) for i in range(len(k1))]
        
        self.P,self.Speed,self.Angle,self.pqr = state
        reward,done = reward_fn(self,state)
        
        return state,reward,done
    
    def reset(self):
#        self.set_acceleration()
        self.set_angle()
        self.set_pqr()
#        self.set_dpqr()
        self.set_speed()
        self.set_Force()
        self.set_location()
        
        state = [self.P,self.Speed,self.Angle,self.pqr]
        return state




#class sim_Window(ShowBase):
#    def __init__(self,mode='explore'):
#        ShowBase.__init__(self)
#        self.model = loader.load_model('3dmodels/mq27.egg')
#        self.model.reparentTo(render)
#        self.agent = quard_Copter()
#        P,_,Angle,_ = self.agent.reset()
#        self.scale = LVector3(0.01,0.01,0.01)
#        self.model.setScale(self.scale)
#        P = LVector3(tuple(P))
#        Angle = LVector3(tuple(Angle/np.pi*180.))
#        self.model.setPos(P)
#        self.model.setHpr(Angle)
#        if mode == 'explore':            
#            self.gameTask = taskMgr.add(self.simLoop_explore, "simLoop_explore")
#            self.OU = utils.OU_process(self.agent.action_dim,self.agent.dt)
#            self.OU.reset()
#    
#    
#    def simLoop_explore(self,task):
#        P,Speed,Angle,pqr = self.agent.P,self.agent.Speed,self.agent.Angle,self.agent.pqr
#        action = self.get_action_explore(np.hstack((P,Speed,Angle,pqr)))
#        state,reward,done = self.agent.step(action)
#        if done:
#            state = self.agent.reset()
#            self.OU.reset()
#        P,Speed,Angle,pqr = state
#        P = LVector3(tuple(P))
#        Angle = LVector3(tuple(Angle/np.pi*180.))
#        self.model.setPos(P)
#        self.model.setHpr(Angle)
#        return task.cont
#    
#    def get_action_explore(self,state):
#        action = pi(torch.from_numpy(state).float().to(device)).data.cpu().numpy()
#        action += self.OU.sample()*self.agent.action_lim
#        return action
#
#a = sim_Window()
#a.run()

