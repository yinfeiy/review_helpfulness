import os, subprocess

genre = 'home'
for feature in ['serv', 'func', 'appe', 'o', 'qual', 'use', 'price', 'brand', 'ovrl']:
    # ['home', 'watches', 'electronis', 'outdoor']
    cmd = 'python fusion_l1.py {0} {1}'.format(feature, genre)
    proc = subprocess.Popen(cmd.split())
    proc.wait()
    print proc.returncode
