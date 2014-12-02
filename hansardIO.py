# -*- coding: utf-8 -*-
"""
hansardIO

We are going to start with the 16th Parliament here (1926) for reasons of
simplicity and ignore earlier data.

"""

from bs4 import BeautifulSoup
import urllib.request
import re
import datetime
import time

class choppedHansardParse:
    ''' Parses an individual 
    '''
    
    def __init__(self):
        print('Mission STUURURRRT')
        self.yearList = range(1867,2000)
        self.dataDir = "chopped"
    
    def lineGenerator(self, file):
        f = open(file)
        for line in iter(f):
            # preprocessor
            yield line
        f.close()
        
    def objectParse(self):
        pass
    
    def parseAll(self):
        '''Generator yields'''
        pass
    
    
class hansardCSV:
    '''Parses and outputs'''
    
    
def parlSessGenerator():
    ''' Scrapes from 
    http://www.parl.gc.ca/parlinfo/compilations/parliament/Sessions.aspx
    for information for ParlSess table'''
    
    response = urllib.request.urlopen(
        'http://www.parl.gc.ca/parlinfo/compilations/parliament/Sessions.aspx')
    html = response.read()
    soup = BeautifulSoup(html)
    
    # create a master dict of parliament urls
    
    urls={}
    for ref in soup.find_all("a", text=(re.compile("\s\sParliament"))):
        for parl in (ref.contents):
            parlInt = int(re.sub("\D", "",str(parl)))
            
        # HERE is where we skip over looking at pre-1926 data
        # Fixing this will require modifications to handle different parties
        # in government/different government statuses within the same
        # parliament and session to handle the 2nd and 15th parliaments
        
        if parlInt < 16:
            pass
        else:
            urls[parlInt]=("http://www.parl.gc.ca/parlinfo/"+(ref.get("href")[6:]))
        
    # load a supplementary csv with wikipedia compiled data
    # much easier than parsing it
    
    
    
    # parse list of parliament urls
    
    for parlNum in list(urls.keys()):
        print (parlNum)
        
        time.sleep(10) # flood delay
        response = urllib.request.urlopen(urls[parlNum])
        html = response.read()
        soup = BeautifulSoup(html)
        
        # term length to two dates
        # election date to a date
        
        TermData=str((soup.find(id="ctl00_cphContent_TermData")).contents[0])
        
        TermStart = datetime.date(int(TermData[:4]), int(TermData[5:7]), 
                                  int(TermData[8:10]))
        TermEnd = datetime.date(int(TermData[13:17]), int(TermData[18:20]), 
                                int(TermData[21:23]))
        ElectionData = str((soup.find(
                id="ctl00_cphContent_DateGeneralElectionData")).contents[0])
        ElectionDate = datetime.date(int(
            ElectionData[:4]), int(ElectionData[5:7]), int(ElectionData[8:10]))
        
        # duration to an int of days
        
        DurationData=str((soup.find(
            id="ctl00_cphContent_DurationData")).contents[0])
        DurationData = "".join(DurationData.split())
        DurationData = (re.sub('\([^)]*\)', '', DurationData)).replace("days","")
        DurationDays = int(DurationData)

        # government type and governing party        
        
        GovernmentType = str((soup.find(
            id="ctl00_cphContent_GovernmentTypeData")).contents[0])
        GoverningParty = str((soup.find(
            id="ctl00_cphContent_GoverningPartyData")).contents[0])
        
        # sessions
        
        NumberSessions = int(str((soup.find(
            id="ctl00_cphContent_NumberSessionData")).contents[0]))

        # Todo: Sessions Table: id ctl00_cphContent_ctl00_grdSessionList
        # Tables tables tables!



        

        
        
        
    
    

    