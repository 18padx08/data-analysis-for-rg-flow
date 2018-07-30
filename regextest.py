# coding=utf8
# the above tag defines encoding for this document and is for Python 2.x compatibility

import re

regex = r"-?0\.\d*(?=(-?0\.|,))"

test_str = "0.0001612690.0946407-0.0009402760.1388420.05487630.0928264,0.02138570.0847117,0.0334361"

matches = re.finditer(regex, test_str, re.MULTILINE)

for matchNum, match in enumerate(matches):
    matchNum = matchNum + 1
    
    print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
    
    for groupNum in range(0, len(match.groups())):
        groupNum = groupNum + 1
        
        print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))

# Note: for Python 2.7 compatibility, use ur"" to prefix the regex and u"" to prefix the test string and substitution.
