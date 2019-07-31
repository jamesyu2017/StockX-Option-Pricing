import numpy as np
import re
import statistics
from operator import add
from pyspark import SparkContext

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: stockX <file> <output> ", file=sys.stderr)
    #     exit(-1)
    def getStat(data):
        mean = statistics.mean(data)
        std = statistics.stdev(data)
        return mean, std

    sc = SparkContext(appName="stockX")
    lines = sc.textFile("StockXData.csv")
    # set header as the first line of data
    header = lines.first()
    # remove header from our data
    lines = lines.filter(lambda row: row != header)
    # split line and remove invalid data
    validLines = lines.map(lambda x: x.split(",")).filter(lambda x: len(x) == 8)
    # print(validLines.first())
    regex = re.compile('[\W_]+')
    mainData = validLines.map(lambda x: (x[0], x[2], float(regex.sub('', x[3])), float(regex.sub(' ', x[4]))))
    # print(mainData.first())
    totalProfit = mainData.map(lambda x: ((x[0], x[1]), float(0.9 * x[2]-x[3]))).groupByKey()
    # get profit for each kind of shoes
    avgProfit = totalProfit.mapValues(lambda x: (sum(x)/len(x))).map(lambda x: (x[0][1], x[1]))
    # print(avgProfit.take(1))
    profVec = avgProfit.groupByKey()
    # print(profVec.take(1))
    shoeStat = profVec.map(lambda x: (x[0], getStat(x[1]))).map(lambda x: (x[0], x[1][0], x[1][1]/x[1][0]))
    # print(shoeStat.take(1))
    sampleStat = shoeStat.take(1)
    print(sampleStat)
    sc.stop()
