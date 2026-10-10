"""Parse exact generated arguments without creating a model or running a benchmark."""
import argparse
import json
import runpy
import sys

commands=json.loads(sys.stdin.read())
original=argparse.ArgumentParser.parse_args
class Parsed(Exception): pass

def parse(self,*args,**kwargs):
    original(self,*args,**kwargs)
    raise Parsed()

argparse.ArgumentParser.parse_args=parse
for command in commands:
    sys.argv=command
    try:
        runpy.run_path(command[0],run_name='__main__')
    except Parsed:
        continue
    raise RuntimeError('Runner did not use its expected parser: '+command[0])
print(json.dumps({'parsed_commands':len(commands)}))
