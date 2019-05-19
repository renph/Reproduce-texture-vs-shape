from pynput.mouse import Button, Controller
from pynput.keyboard import Controller as keyController
from pynput.keyboard import Key
import time
mouse = Controller()
keyboard = keyController()
print('start')
def alttab():
    keyboard.press(Key.alt)
    keyboard.press(Key.tab)
    keyboard.release(Key.tab)
    keyboard.release(Key.alt)

def forwardtabs():
    keyboard.press(Key.ctrl)
    keyboard.press(Key.tab)
    keyboard.release(Key.tab)
    keyboard.release(Key.ctrl)

def backwardtabs():
    keyboard.press(Key.ctrl)
    keyboard.press(Key.shift)
    keyboard.press(Key.tab)
    keyboard.release(Key.tab)
    keyboard.release(Key.shift)
    keyboard.release(Key.ctrl)


time.sleep(5)
while True:
    try:
        # alttab()
        time.sleep(10)

        mouse.scroll(0, 20)
        time.sleep(2)
        mouse.scroll(0, -200)
        time.sleep(2)

        forwardtabs()
        time.sleep(2)
        backwardtabs()
        # alttab()
    except Exception as e:
        print(e)
        break