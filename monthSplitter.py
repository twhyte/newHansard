# -*- coding: utf-8 -*-
"""
monthSplitter.py

Created on Sat Feb  7 15:24:33 2015

@author: twhyte

This script reconfigures dilipad Canadian Hansard text files for 
sentiwordnet analysis etc.
First the original files are cleaned to repair filenames and flag errors for
manual correction.
Second, a new directory amalgamates data according to year/month.

We begin with Feb 6 1901 (9th Parliament)

"""

import os
import re
import copy
import calendar

### some sorting code for filenames with numbers

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


oldFolder = "chopped" # source data directory
newFolder = "monthSplit"

if not os.path.exists(os.path.join(os.getcwd(),newFolder)):
    os.makedirs(newFolder)
dirs = os.listdir(os.path.join(os.getcwd(),oldFolder))

# clean unwanted dirs from dirlist

for i in dirs[:]:
    if i.startswith("ooci"):
        dirs.remove(i)
    for parl in range(1,9):
        if i.startswith(("HOC0"+str(parl))):
            dirs.remove(i)

sort_nicely(dirs)

for workingDir in dirs:

    fileList = os.listdir(os.path.join(oldFolder,workingDir))
    sort_nicely(fileList)
    
    filename_month=""
    filename_year=""
    
    for file in fileList:
        # test that this is not an index file
    
        if "dump" in file:
            pass
        elif file[0:3].isdigit()==True:
            pass
        elif "Government" in file:
            pass
        elif "on_" in file:
            pass
        elif "__" in file:
            pass
        elif "Hawkes" in file:
            pass
        elif "SO_31" in file:
            pass
        
        
        else:
        
            # this is a valid text file
            # first clean common filename errors in our old files
        
            
            crap = ["th", ".txt", "nd", "st"]
            for t in crap:
                if t in file:
                    oldName = copy.deepcopy(file)
                    newName=re.sub(t, "", file)
                    os.rename(os.path.join(oldFolder,workingDir,oldName),
                              os.path.join(oldFolder,workingDir,newName))
                    file=newName
                    

            if (file[0]).isdigit():
                # some files begin with the day--
                # we assume this is a mistake and try to correct it,
                # but on FileExistsError break since it will need to be
                # manually merged with an existing file
            
                filename_month = re.split('_', file)[1]
                filename_year = re.split('_', file)[2]
                filename_day = re.split('_', file)[0]
                oldName=copy.deepcopy(file)
                newName=(filename_month+"_"+filename_day+"_"+filename_year)
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                          os.path.join(oldFolder,workingDir,newName))
                file=newName
     
            try:
                filename_month = re.split('_', file)[0]
                filename_year = re.split('_', file)[2]
                
            except IndexError: 
            # some filenames are missing one underscore, fix these
            # again, a persistent exception means manual correction is necessary
            
                try:
                    filename_month = re.split('_', file)[0]
                    filename_year = re.split('_', file)[1][-4:]
                    filename_day = re.findall(".*?[_](.*)[\d]{4}\Z", file)[0]
                    oldName = copy.deepcopy(file)
                    file = filename_month+"_"+filename_day+"_"+filename_year

                    
                    os.rename(os.path.join(oldFolder,workingDir,oldName),
                              os.path.join(oldFolder,workingDir,file))
              
                except TypeError:
                    print("We failed at this filename in this dir:")
                    print(file)
                    print(workingDir)
                    
                except FileExistsError: 
               
                    pass


            # refresh this info for safety
            
            filename_month = re.split('_', file)[0]
            filename_year = re.split('_', file)[2]
            filename_day = re.split('_', file)[1]
            
                
            # batch correct some of these
                
            if "3" in filename_year[1]:
                oldName=copy.deepcopy(file)
                oldYear=copy.deepcopy(filename_year)
                filename_year="19"+oldYear[2:4]
                file = filename_month+"_"+filename_day+"_"+filename_year
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                os.path.join(oldFolder,workingDir,file))
            
            if filename_month in ["Augu","Augusl"]:
                filename_month="August"
                oldName=copy.deepcopy(file)
                file = filename_month+"_"+filename_day+"_"+filename_year
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                os.path.join(oldFolder,workingDir,file))
            if filename_month in ["Feburary","Februry", "Feb"]:
                filename_month="February"
                oldName=copy.deepcopy(file)
                file = filename_month+"_"+filename_day+"_"+filename_year
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                os.path.join(oldFolder,workingDir,file))
            if filename_month in ["ApriJ","Apxil"]:
                filename_month="April"
                oldName=copy.deepcopy(file)
                file = filename_month+"_"+filename_day+"_"+filename_year
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                os.path.join(oldFolder,workingDir,file))
            if filename_month in ["Jan"]:
                filename_month="January"
                oldName=copy.deepcopy(file)
                file = filename_month+"_"+filename_day+"_"+filename_year
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                os.path.join(oldFolder,workingDir,file))
            if filename_month in ["Jidy","Jnly"]:
                filename_month="July"
                oldName=copy.deepcopy(file)
                file = filename_month+"_"+filename_day+"_"+filename_year
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                os.path.join(oldFolder,workingDir,file))
            if filename_month in ["Ociober","Oclober"]:
                filename_month="July"
                oldName=copy.deepcopy(file)
                file = filename_month+"_"+filename_day+"_"+filename_year
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                os.path.join(oldFolder,workingDir,file))
            if filename_month in ["Marclf"]:
                filename_month="March"
                oldName=copy.deepcopy(file)
                file = filename_month+"_"+filename_day+"_"+filename_year
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                os.path.join(oldFolder,workingDir,file))
            if filename_month in ["Seplember"]:
                filename_month="September"
                oldName=copy.deepcopy(file)
                file = filename_month+"_"+filename_day+"_"+filename_year
                os.rename(os.path.join(oldFolder,workingDir,oldName),
                os.path.join(oldFolder,workingDir,file))
                
            if "9" not in filename_year[1]: # troubleshooting help for bad names
                print ((os.path.join(oldFolder,workingDir,file)))
            if filename_month not in list(calendar.month_name)[1:]:
                print ((os.path.join(oldFolder,workingDir,file)))
            if "day" in filename_year: # troubleshooting help for bad names
                print ((os.path.join(oldFolder,workingDir,file)))
            
            # okay finally start on the file text
            
            f = open((os.path.join(oldFolder,workingDir,file)), "r", encoding="utf8")
            lines = f.readlines()

################
#            # here we search for dates that haven't been excised
#            # dunno if this code will become necessary
#           
#            heldDates = [] 
#
#            
#            for line in lines:
#                yearSearch = re.findall("[A-Z]+\s[\d]+[,|.]\s19\d{2}", file)
#                if len(yearSearch) !=0:
#                    for x in list(range(len(yearSearch))):
#                        heldDates.append(x)
#################                
            
            # NOW we start amalgamating files to monthly units
            # first check if a file exists for the current working year and month
            # if it's empty, this is the beginning of the new file
            # if it's not empty, then we want to append to the end
            
            # first, check that if we are in a new or existing year
            
            if not os.path.exists(os.path.join(os.getcwd(),newFolder,filename_year)):
                os.makedirs(os.path.join(os.getcwd(),newFolder,filename_year))

            newFile = filename_month+"_"+filename_year
            
            try: #Write new or append existing file
            
                w = open(os.path.join(os.getcwd(),newFolder,filename_year,newFile), 'a', encoding="utf8")
                w.write("\n")
                for line in lines:
                    w.write(line)
                w.close()
                f.close()
                
            except:
                raise Exception #something weird has happened
            
            
            
            
            
            
                        

            
    