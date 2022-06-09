import multiprocessing as mp
import time

import detector
import mail

from settings import *

send_mail    = mp.Event()

detect_proc = mp.Process(target=detector.loop, args=(send_mail, ), daemon=True)
mail_proc   = mp.Process(target=mail.loop, args=(send_mail, ), daemon=True)

if __name__ == "__main__":
    try:
        # Initialization
        send_mail.clear()
        detect_proc.start()
        mail_proc.start()
        # Infinite loop
        while True:
            time.sleep(1)
    except:
        # Check if any process is still running
        while detect_proc.is_alive() or mail_proc.is_alive():
            time.sleep(1)
        # Closing processes
        detect_proc.close()
        mail_proc.close()
        # Exit program
        exit_time = time.strftime("%Y/%b/%d-%H:%M:%S", time.localtime())
        print("(" + exit_time + ") EXIT")
        exit()
