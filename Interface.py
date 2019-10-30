import wx


def yes_no_button(panel):
    yes_button = wx.Button(panel, wx.ID_YES)
    yes_button.Bind(wx.EVT_BUTTON, panel.on_yes_press)

    no_button = wx.Button(panel, wx.ID_NO)
    no_button.Bind(wx.EVT_BUTTON, panel.on_no_press)

    button_sizer = wx.BoxSizer(wx.HORIZONTAL)
    button_sizer.Add(yes_button)
    button_sizer.Add((20, 0))
    button_sizer.Add(no_button)

    return button_sizer


def panel_text(panel):
    my_text = wx.StaticText(panel, 0, style=wx.ALIGN_CENTER_HORIZONTAL)
    font = wx.Font(12, wx.FONTFAMILY_MODERN, wx.NORMAL, wx.NORMAL)
    my_text.SetFont(font)
    my_text.SetLabel("Do you want to download the\ntraining dataset?")

    return my_text


class MyPanel(wx.Panel):

    def __init__(self, frame):
        super().__init__(frame)

        self.frame = frame
        buttons = yes_no_button(self)
        text = panel_text(self)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(text, 0, wx.ALL | wx.CENTER, 30)
        main_sizer.Add(buttons, 0, wx.ALL | wx.CENTER, 10)
        self.SetSizer(main_sizer)
        self.Fit()

    def on_yes_press(self, event):
        self.frame.download = True
        self.frame.Close()

    def on_no_press(self, event):
        self.frame.download = False
        self.frame.Close()


class DownloadButton(wx.Frame):

    def __init__(self):
        super().__init__(parent=None, title='Neural Network')

        self.download = None
        self.panel = MyPanel(self)
        self.Fit()
        self.Center()
        self.Show()
