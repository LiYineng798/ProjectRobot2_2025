k210_models.initialization()
music.set_built_in_speaker_enabled(True)
face = -2
basic.clear_screen()

def on_forever():
    if face == 0:
        music._play_default_background(music.built_in_playable_melody(Melodies.DADADADUM),
            music.PlaybackMode.UNTIL_DONE)
        basic.clear_screen()
    if face == 1:
        music._play_default_background(music.built_in_playable_melody(Melodies.ENTERTAINER),
            music.PlaybackMode.UNTIL_DONE)
        basic.clear_screen()
    if face == 1:
        music._play_default_background(music.built_in_playable_melody(Melodies.PRELUDE),
            music.PlaybackMode.UNTIL_DONE)
        basic.clear_screen()
basic.forever(on_forever)

def on_forever2():
    global face
    face = k210_models.face_reg()
    if face >= 0:
        basic.show_string("" + str((face)))
basic.forever(on_forever2)
