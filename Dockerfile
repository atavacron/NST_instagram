#Improvement: try a lighter version of python
FROM python:3.6

#instead of creating a workdir, just add necessary files
COPY requirements.txt ./

ADD neural_style_transfer.py ./

#you might have a problem with accepting the app add the pickle
#################WARNING################################
#If so DO NOT publish the container
#################WARNING################################
ADD google_api.py ./

ADD credentials.json ./

#do not read unless you have in your env the token pickle
ADD token.pickle ./

ADD style_name.json ./

#even though logs would not store anything, script will brake if not found
ADD logs.csv ./

RUN pip install -r requirements.txt

#necessary folders
RUN mkdir output

RUN mkdir output/output_log

RUN mkdir prod_folder

#logic is no longer in the bash file
CMD [ "python", "./neural_style_transfer.py" ]