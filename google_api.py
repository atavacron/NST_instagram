from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
from apiclient import errors
import random
import io

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

def images_extraction(service, id_folder, list_files = False):
    """
    This will print the files in the specified folder. Making sure the ones you want.
    Parameters
    ----------
    service : credentials of Google Drive API
    id_folder : ID of your folder. Can be found in the web page link
    """
    images = []
    page_token = None
    while True:
        #listing all files in folder id
        response = service.files().list(q="'{}' in parents".format(id_folder),
                                              spaces='drive',
                                              fields='nextPageToken, files(id, name)',
                                              pageToken=page_token).execute()
        for file in response.get('files', []):
            if list_files:
                #in case list files true then will print all your files and will not store data
                print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
            else:
                #list of list, containing id (to download) and name (for json file)
                images.append([file.get("id"), file.get("name")])
        age_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    #will select only one image
    selected = random.choice(images)
    return(selected)

def download_images(service, file_id, path):
    """
    This will download the image of file_id into prod_folder and name content.jpg. 
    Take into account should only be jpg images.
    Parameters
    ----------
    service : credentials of Google Drive API
    file_id : ID of your image.
    """
    request = service.files().get_media(fileId=file_id)
    #if you want to save in other folder change here
    fh = io.FileIO('prod_folder/' + path, 'wb')
    #BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))
    return(fh.seek(0))


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    #check if I have tocken.pickle file, if so load the credentials If not valid refresh
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next time
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    path_content = "your folder content id on google drive"
    path_style = "your folder style id on google drive"
    image_content = images_extraction(service, path_content)
    image_style = images_extraction(service, path_style)
    downloader_content = download_images(service, image_content[0], "content.jpg")
    downloader_style = download_images(service, image_style[0], "style.jpg")
    return(image_style[1], image_content[1])

if __name__ == '__main__':
    main()
