from main_clean import highlight_keywords, count_keywords

samples = [
    ". Supervised",
    ", UNIX,",
    "Java,",
    "Skills: Python, C#, C ++.",
    "C # is sometimes written with spaces.",
    "Worked on node.js and Docker.",
    "Implemented C++ and C+++ examples.",
    "Experience with c++ and c# and node.js",
    "Leading teams: supervised, mentored, and coached.",
]

for s in samples:
    print('INPUT:', repr(s))
    print('HIGHLIGHT:', highlight_keywords(s))
    print('COUNTS (h,s,r):', count_keywords(s))
    print('---')
