import sys
import datetime

time = datetime.datetime.now()

output = "testing %s the score is %s" % (sys.argv[1], time)

print(output)
