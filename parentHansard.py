# -*- coding: utf-8 -*-
"""
parentHansard
"""

import datetime

class hansardFormatting:
    '''establishes HansardFormatting type event class'''
    
    def __init__(self):
        self.FormattingID = int()
        self.FormattingType = str()
        self.FormattingText = str()
        
class hansardProcedural:
    '''establishes HansardProcedural type event class'''
    
    def __init__(self):
        self.ProceduralID = int()
        self.ProceduralType = str()
        self.ProceduralText = str()
        
class hansardStatement:
    '''establishes HansardStatement type event class'''
    
    def __init__(self):
        self.StatementID = int()
        self.StatementOwnerID = int()
        self.StatementText = str()

class hansardEvent:
    '''establishes generic Hansard event class'''
    
    def __init__(self):
        self.HansardEventID = int()
        self.HansardObjectID = int()
        self.HansardEventType = str()
        self.HansardEventContent = ()
        
class HansardObject:
    '''establishes generic Hansard object (Hansard Day) class'''
    
    def __init__(self):
        self.HansardObjectID = int()
        self.HansardDate = datetime.date()
        self.ParlSessID = int()
        
class ParlSess:
    '''establishes ParlSess class'''
    
    def __init__(self):
        self.ParlSessID = int()
        self.ParliamentCount = int()
        self.ParliamentStartDate = datetime.date()
        self.ParliamentEndDate = datetime.date()
        self.ParliamentElectionDate = datetime.date()
        self.ParliamentTotalSessions = int()
        self.PartyGovernment = str()
        self.PartyOpposition = str()
        self.GovernmentType = str()
        self.GovernmentSeatsFrac = float()
        self.NumberOfficialParties = int()  
        self.ParliamentSession = int()
        self.ParliamentSessionSittingDays = int()
        self.ParliamentSessionStartDate = datetime.date()
        self.ParliamentSessionEndDate = datetime.date()
        
        
           
    
    
        
        
        
    