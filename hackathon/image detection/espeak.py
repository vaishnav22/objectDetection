data = input("enter the eext\n")
devnull = open('os,devnull','w')
subprocess.run(["espeak","-k5","-s150",'data'],stdout=devnull,stderr=subprocess.STDOUT)