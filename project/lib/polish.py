import csv, readline, re, sys, math, uuid

def main():
    """ Array of Arrays. """
    rides = []
    key = 0
    """ Opening CSV file, need to encode to UTF-8. """
    with open('random_rides_200k.csv', 'r',encoding='utf-8') as source:
        reader = csv.reader(source)
        """ Skipped the first line because it was the header. """
        header = reader.__next__()
        reader.__next__()
        rides.append(header)
        print(header)
        counter = 0
        for item in header:
            print("{0}: {1}".format(counter,item))
            counter += 1
        for row in reader:
            try:
                if 0 in [float(row[6]),float(row[7]),float(row[8]),float(row[9])]:
                    continue
            except:
                pass
            rides.append(row)
    target = open('random_rides_200k_edited.csv', 'wt',encoding='utf-8')
    try:
        writer = csv.writer(target,dialect='excel')
        for row in rides:
            writer.writerow(row)
    finally:
        target.close()
main()
