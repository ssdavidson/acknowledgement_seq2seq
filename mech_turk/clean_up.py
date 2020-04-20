import csv, sys

csv_reader = csv.reader(open(sys.argv[1]), delimiter=',')
csv_writer = csv.writer(open(sys.argv[1][:-3] + 'clean.csv', mode = 'w+'), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

for line in csv_reader:
    context = line[0]
    input = line[1]
    result = line[2]

    result = result.replace(" me are", " I'm")
    result = result.replace(" me 're", " I'm")
    result = result.replace(" I 're", " I'm")
    result = result.replace(" do me", " I")
    result = result.replace(" do you", " you")
    result = result.replace(" you 'd", "you'd")
    result = result.replace(" you 're", "you're")
    result = result.replace(" 's", " is")
    result = result.replace(" 'm", " am")
    result = result.replace(" 're", " are")
    result = result.replace(" 've", " have")
    result = result.replace(" 'd", " would")
    result = result.replace(" 'll", " will")
    result = result.replace(" na ", "na ")
    result = result.replace(" n't", "n't")

    csv_writer.writerow([context,input,result])
