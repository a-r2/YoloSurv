import os, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from settings import *

# Create a multipart message and set headers
message            = MIMEMultipart()
message["From"]    = SENDER_EMAIL
message["To"]      = RECEIVER_EMAIL
message["Subject"] = SUBJECT

def send_mails():
    # Find video files in directory
    dirls = os.listdir(DIRPATH)
    # Select video file
    mp4paths = list()
    for item in dirls:
        filename, filext = item.split(".")
        if filext == "mp4":
            filepath = "/".join((DIRPATH, item))
            mp4paths.append(filepath)
    if len(mp4paths) > 0:
        for mp4path in mp4paths:
            # Open video file in binary mode
            filename = mp4path.split("/")[-1]
            with open(mp4path, "rb") as attachment:
                # Add file as application/octet-stream
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            # Encode file in ASCII characters to send by email    
            encoders.encode_base64(part)
            # Add header as key/value pair to attachment part
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={filename}",
            )
            # Add attachment to message and convert message to string
            message.attach(part)
            text = message.as_string()
            # Log in to server using secure context and send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(SENDER_EMAIL, APP_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
                server.close()
            # Remove mp4 file
            os.remove(mp4path)
            message.set_payload(None)

def loop(send_mail):
    try:
        while True:
            # Wait for sending mail flag
            send_mail.wait()
            # Send mails
            send_mails()
            # Clear sending mail flag
            send_mail.clear()
    except:
        # Wait for sending mail flag
        send_mail.wait()
        # Send mails
        send_mails()
        # Print message
        print("MAIL PROCESS ENDED SUCCESSFULLY!")
