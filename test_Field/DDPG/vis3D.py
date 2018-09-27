#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 01:58:41 2018

@author: ZhangYaoZhong
"""

from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.distributed.PyDatagram import PyDatagram
from direct.distributed.PyDatagramIterator import PyDatagramIterator
import numpy as np

class netCommon:
    def __init__(self,protocal):
        self.manager = ConnectionManager()
        self.reader = QueuedConnectionReader(self.manager,0)
        self.writer = ConnectionWriter(self.manager,0)
        self.protocal = protocal
        taskMgr.add(self.updateReader,'updateReader')
    
    def updateReader(self,task):
        if self.reader.dataAvailable():
            data = NetDatagram()
            self.reader.getData(data)
            reply = self.protocal.process(data)
            
            if reply != None:
                self.writer.send(reply,data.getConnection())
        return task.cont

class Vis3D_Server(netCommon):
    def __init__(self,protocal,port):
        netCommon.__init__(self,protocal)
        self.listener = QueuedConnectionListener(self.manager,0)
        socket = self.manager.openTCPServerRendezvous(port,100)
        self.listener.addConnection(socket)
        self.connetions = []
        
        taskMgr.add(self.updateListener,'updateListener')
    
    def updateListener(self,task):
        if self.listener.newConnectionAvailable():
            connection = PointerToConnection()
            if self.listener.getNewConnection(connection):
                connection = connection.p()
                self.connetions.append(connection)
                self.reader.addConnection(connection)
                print('connection established')
                
        return task.cont

class Vis3D_Client(netCommon):
    def __init__(self,protocal):
        netCommon.__init__(self,protocal)
        
    def connect(self,host,port,timeout):
        self.connection = self.manager.openTCPClientConnection(host,port,timeout)
        if self.connection:
            self.reader.addConnection(self.connection)
            print('client connect to server')
            
    def send(self,datagram):
        if self.connection:
            self.writer.send(datagram,self.connection)

class Protocal:
    def process(self,data):
        return None
    
    def buildReply(self,msgid,data):
        reply = PyDatagram()
        reply.addUnit8(msgid)
        reply.addString(data)
        return reply

class ServerProtocal(Protocal):
    def process(self,data):
        it = PyDatagramIterator(data)
        msgid = it.getUnit8()
        

class ClientProtocal(Protocal):
    def process(self,data):
        it = PyDatagramIterator(data)
        msgid = it.getUnit8()
    
    

class Vis3D_App(ShowBase):
    def __init__(self,env,model_name):
        ShowBase.__init__(self)
        
#        sever = Vis3D_Server(SeverProtocal,9999)
#        client = Vis3D_Client(ClientProtocal)
#        client.connect('localhost',9999,3000)
        
        
        model_path = '3dmodels/'+model_name
        self.model = loader.load_model(model_path)
        self.model.reparentTo(render)
        self.env = env
#        data = PyDatagram()
        
        taskMgr.add(self.updateState,'updateState')
    
    def updateState(self,task):
        
        P_sim = self.env.P
        Angle_sim = self.env.Angle
        P = LVector3(tuple(P_sim))
        Angle = LVector3(tuple(Angle_sim/np.pi*180.))
        self.model.setPos(P)
        self.model.setHpr(Angle)
        
        return task.cont
        
        
#test = Vis3D_App('mq27.egg')
#test.run()
#class Vis3D_Client(ShowBase):
#    def __init__(self,model_name):
#        ShowBase.__init__(self)
#        model_path = '3dmodels/'+model_name
#        self.model = loader.load_model(model_path)
#        self.model.reparentTo(render)
#        self.http = HTTPClient.getGlobalPtr()
#        self.channel = self.http.makeChannel(False)
#        self.ramfile = Ramfile()
#        taskMgr.add(self.updateChannel,'updateChannel')
#        
        
#class Sim_3DQC(ShowBase):
#    def __init__(self,pi,mode='explore'):
#        ShowBase.__init__(self)
#        self.model = loader.load_model('3dmodels/mq27.egg')
#        self.model.reparentTo(render)
#        self.env = qc.quard_Copter()
#        self.action_dim = self.env.action_dim
#        self.action_lim = self.env.action_lim
#        self.state_dim = self.env.state_dim
#        P,_,Angle,_ = self.agent.reset()
#        self.scale = LVector3(0.01,0.01,0.01)
#        self.model.setScale(self.scale)
#        P = LVector3(tuple(P_sim))
#        Angle = LVector3(tuple(Angle_sim/np.pi*180.))
#        self.model.setPos(P)
#        self.model.setHpr(Angle)
#        self.pi = pi
#        if mode == 'explore':            
#            self.gameTask = taskMgr.add(self.sim_3Dexplore, "simLoop_explore")
#            self.OU = utils.OU_process(self.agent.action_dim,self.agent.dt)
#            self.OU.reset() 
#        
#    def sim_3Dexplore(self,task):
#        P,Speed,Angle,pqr = self.env.P,self.env.Speed,self.env.Angle,self.env.pqr
#        action = self.get_action_explore(np.hstack((P,Speed,Angle,pqr)))
#        state,reward,done = self.env.step(action)
#        if done:
#            state = self.env.reset()
#            self.OU.reset()
#        P,Speed,Angle,pqr = state
#        P = LVector3(tuple(P))
#        Angle = LVector3(tuple(Angle/np.pi*180.))
#        self.model.setPos(P)
#        self.model.setHpr(Angle)
#        return task.cont
#    
#    def get_action_explore(self,state):
#        action = self.pi(torch.from_numpy(state).float().to(device)).data.cpu().numpy()
#        action += self.OU.sample()*self.agent.action_lim
#        return action