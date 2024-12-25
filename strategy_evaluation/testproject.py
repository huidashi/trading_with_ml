import marketsimcode as ms
import datetime as dt
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import indicators as indi
import numpy as np
import subprocess
import ManualStrategy

def author():
  return 'hshi320'


if __name__ == '__main__':
    subprocess.run(["python", "ManualStrategy.py"])
    subprocess.run(["python", "experiment1.py"])
    subprocess.run(["python", "experiment2.py"])