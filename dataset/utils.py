import random
import os

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageColor


def show_img(img):
    cv2.imshow('name', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Line(object):
    def __init__(self, line_thickness=0, type=1):
        assert line_thickness % (type + 1) == 0
        self.line_thichness = line_thickness
        self.type = type  # 0 là ẩn, n = n vạch

    def set_invisible(self):
        self.type = 0


class Row(object):
    def __init__(self, height, margin_top, margin_bottom, index, line_top: Line, line_bottom: Line):
        self.height = height
        self.index = index
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.line_top = line_top
        self.line_bottom = line_bottom

    def get_height(self):
        return self.height + self.line_top.line_thichness + self.line_bottom.line_thichness


class Col(object):
    def __init__(self, width, margin_left, margin_right, index, line_left: Line, line_right: Line):
        self.width = width
        self.index = index
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.line_left = line_left
        self.line_right = line_right

    def get_width(self):
        return self.width + self.line_left.line_thichness + self.line_right.line_thichness


class Cell(object):

    def __init__(self, col: Col, row: Row, text: str, font: str, align: str, size: int, cell_id: int):
        self.col = col
        self.row = row
        self.text = text
        self.cell_id = cell_id
        self.font = font
        self.align = align
        self.size = size


class Table1(object):
    def __init__(self, widths: list = [0.6, 0.2, 0.2], table_widths=1000, table_height=500,
                 margin_left=10, margin_right=10, margin_top=5, margin_bottom=1):
        self.width_each_cell = list(map(lambda x: int(table_widths * x), widths))
        self.size = map_pixel_to_size(int(41 * table_widths / 1500))
        self.n_rows = table_height // Row(height=int(41 * table_widths / 1500), margin_top=margin_top,
                                          margin_bottom=margin_bottom,
                                          index=-1, line_top=Line(2, 1), line_bottom=Line(6, 2)).get_height()
        self.n_cols = len(widths)
        self.cols = []
        self.rows = []
        self.cells = [[] for _ in range(self.n_rows)]

        # <------------------ gererate text in table --------------------->
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        font_folder = os.path.dirname(cur_dir)
        self.fonts = [f'{font_folder}/font/times.ttf',
                      f'{font_folder}/font/timesbd.ttf',
                      f'{font_folder}/font/timesi.ttf',
                      f'{font_folder}/font/timesbi.ttf']

        # dong` dau tien
        text = [[''] * self.n_cols for _ in range(self.n_rows)]
        for i in range(1, self.n_cols):
            text[0][i] = generate_date_data()
        # dong 2 -> n-1
        for i in range(1, self.n_rows - 1):
            if random.random() < 0.2:
                continue
            text[i][0] = generate_string(random.randint(3, 10))
            for j in range(1, self.n_cols):
                if random.random() < 0.2:
                    continue
                text[i][j] = generate_tien(random.randint(7, 12))
        # dong n
        text[self.n_rows - 1][0] = "Cộng"
        for j in range(1, self.n_cols):
            text[self.n_rows - 1][j] = generate_tien(random.randint(7, 12))
        # <------------------ \gererate text in table --------------------->
        # <------------------ compute height each cell --------------------->
        image_font = ImageFont.truetype(font=self.fonts[1], size=self.size)
        a_t = []
        for t in text:
            a_t += t
        _t = ' '.join(a_t)
        text_height = max([image_font.getsize(w)[1] for w in _t.split()])
        text_height += margin_top + margin_bottom
        self.heigh_each_cell = [text_height] * self.n_rows
        # <------------------ \compute height each cell --------------------->
        #
        for i, w in enumerate(self.width_each_cell[:-1]):
            self.cols.append(Col(width=w, margin_left=margin_left, margin_right=margin_right,
                                 index=i, line_left=Line(2, 1), line_right=Line(0, 0)))
        self.cols.append(Col(width=self.width_each_cell[-1], margin_left=margin_left, margin_right=margin_right,
                             index=len(self.width_each_cell), line_left=Line(2, 1), line_right=Line(2, 1)))

        for i, h in enumerate(self.heigh_each_cell[:-1]):
            self.rows.append(Row(height=h, margin_top=margin_top, margin_bottom=margin_bottom,
                                 index=i, line_top=Line(2, 1), line_bottom=Line(0, 0)))
        self.rows.append(Row(height=self.heigh_each_cell[-1], margin_top=margin_top, margin_bottom=margin_bottom,
                             index=len(self.heigh_each_cell), line_top=Line(2, 1), line_bottom=Line(6, 2)))

        self.table_height = sum([r.get_height() for r in self.rows])
        self.table_width = sum([c.get_width() for c in self.cols])

        for i, r in enumerate(self.rows):
            for j, c in enumerate(self.cols):
                if i == 0 or i == self.n_rows - 1:
                    self.cells[i].append(
                        Cell(col=c, row=r, text=text[i][j], cell_id=j, font=self.fonts[1], align='left',
                             size=self.size))
                else:
                    self.cells[i].append(
                        Cell(col=c, row=r, text=text[i][j], cell_id=j, font=self.fonts[0], align='left',
                             size=self.size))

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if j != 0:
                    self.cells[i][j].align = 'right'

    def get_col_start(self, index):
        return sum([c.get_width() for c in self.cols[:index]])

    def get_row_start(self, index):
        return sum([r.get_height() for r in self.rows[:index]])

    def get_table_bounding_box(self):
        return (self.table_height, self.table_width)

    def draw(self, background_color=255):
        img = np.zeros((self.table_height, self.table_width, 3), dtype=np.uint8)
        img[:, :] = background_color
        for i in range(self.n_rows):
            if i != 1 and i != self.n_rows - 1:
                continue
            for j in range(self.n_cols):
                if i == self.n_rows - 1 and j != 0:
                    img = self.show_cell_by_xy(img, i, j, left=False, right=False, top=True, bottom=True,
                                               margin_left=True, margin_right=True)
                elif i == self.n_rows - 1 and j == 0:
                    continue
                else:
                    img = self.show_cell_by_xy(img, i, j, left=False, right=False, top=True, bottom=True)

        for i_r, row in enumerate(self.rows):
            for i_c, col in enumerate(self.cols):
                img, width = self.draw_text_cell_by_xy(img, i_r, i_c)
        return img

    def draw_line(self, line: Line, img, xmin, xmax, ymin, ymax, orient='vertical'):
        assert orient in ['vertical', 'horizontal']
        if line.type == 0:
            return img
        line_thichness = line.line_thichness // (2 * line.type - 1)
        for i in range(2 * line.type - 1):
            if i % 2 == 0:
                if orient == 'horizontal':
                    start = xmin + i * line_thichness
                    img[start:start + line_thichness, ymin:ymax, :] = 0
                else:
                    start = ymin + i * line_thichness
                    img[xmin:xmax, start:start + line_thichness, :] = 0
        return img

    def show_cell_by_xy(self, img, x: int, y: int, left=False, right=False, top=False, bottom=False,
                        margin_left=False, margin_right=False, margin_top=False, margin_bottom=False):
        xmin = self.get_row_start(x) if not margin_top else self.get_row_start(x) + self.rows[x].margin_top
        xmax = xmin + self.rows[x].get_height() if not margin_bottom else xmin + self.rows[x].get_height() \
                                                                          - self.rows[x].margin_bottom
        ymin = self.get_col_start(y) if not margin_left else self.get_col_start(y) + self.cols[y].margin_left
        ymax = ymin + self.cols[y].get_width() if not margin_right else self.get_col_start(y) + self.cols[y].get_width() \
                                                                        - self.cols[y].margin_right

        if left:
            img = self.draw_line(line=self.cols[y].line_left, img=img, orient="vertical",
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymin + self.cols[y].line_left.line_thichness)
        if right:
            img = self.draw_line(line=self.cols[y].line_right, img=img, orient="vertical",
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymax - self.cols[y].line_right.line_thichness,
                                 ymax=ymax)
        if top:
            img = self.draw_line(line=self.rows[x].line_top, img=img, orient="horizontal",
                                 xmin=xmin,
                                 xmax=xmin + self.rows[x].line_top.line_thichness,
                                 ymin=ymin,
                                 ymax=ymax)
        if bottom:
            img = self.draw_line(line=self.rows[x].line_bottom, img=img, orient="horizontal",
                                 xmin=xmax - self.rows[x].line_bottom.line_thichness,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymax)
        return img

    def draw_text_cell_by_xy(self, img, x: int, y: int, underline=False):
        xmin, ymin, xmax, ymax = self.get_bounding_box_by_xy(x, y)
        img, width = draw_text(text=self.cells[x][y].text, font=self.cells[x][y].font, align=self.cells[x][y].align,
                               size=self.cells[x][y].size,
                               xmin=ymin,
                               ymin=xmin,
                               xmax=ymax, ymax=xmax, img=img, text_color='#000000,#282828', underline=underline)
        return img, width + self.cols[y].margin_right + self.cols[y].line_left.line_thichness

    def get_bounding_box_by_xy(self, x, y):
        # row
        xmin = self.get_row_start(x) + self.rows[x].line_top.line_thichness + self.rows[x].margin_top
        xmax = self.get_row_start(x) + self.rows[x].get_height() \
               - self.rows[x].line_bottom.line_thichness - self.rows[x].margin_bottom
        # col
        ymin = self.get_col_start(y) + self.cols[y].line_left.line_thichness + self.cols[y].margin_left
        ymax = self.get_col_start(y) + self.cols[y].get_width() \
               - self.cols[y].line_right.line_thichness - self.cols[y].margin_right
        return xmin, ymin, xmax, ymax


class Table2(object):
    def __init__(self, widths: list = [0.5, 0.4, 0.1], table_widths=1000, table_height=500,
                 margin_left=10, margin_right=10, margin_top=5, margin_bottom=1):
        self.width_each_cell = list(map(lambda x: int(table_widths * x), widths))
        self.size = map_pixel_to_size(int(41 * table_widths / 1500))
        self.n_rows = table_height // Row(height=int(41 * table_widths / 1500), margin_top=margin_top,
                                          margin_bottom=margin_bottom,
                                          index=-1, line_top=Line(2, 1), line_bottom=Line(6, 2)).get_height()
        self.n_cols = len(widths)
        self.cols = []
        self.rows = []
        self.cells = [[] for _ in range(self.n_rows)]

        # <------------------ gererate text in table --------------------->
        font_folder = os.path.dirname(os.path.abspath(__file__))
        font_folder = os.path.dirname(font_folder)
        self.fonts = [f'{font_folder}/font/times.ttf',
                      f'{font_folder}/font/timesbd.ttf',
                      f'{font_folder}/font/timesi.ttf',
                      f'{font_folder}/font/timesbi.ttf']
        # dong` dau tien
        text = [[''] * self.n_cols for _ in range(self.n_rows)]
        for i in range(0, self.n_cols - 1):
            text[0][i] = generate_string(random.randint(3, 10))
        # dong 2 -> n-1
        for i in range(1, self.n_rows-1):
            if random.random() < 0.1:
                continue
            text[i][0] = generate_string(random.randint(3, 10))
            for j in range(1, self.n_cols - 1):
                if random.random() < 0.2:
                    continue
                if random.random() < 0.5:
                    text[i][j] = generate_rate()
                else:
                    text[i][j] = generate_string(random.randint(3, 8))
        # dong n
        for i in range(0, self.n_cols - 1):
            if random.random() < 0.5:
                text[self.n_rows-1][i] = generate_rate()
            else:
                text[self.n_rows-1][i] = generate_string(random.randint(3, 8))
        # <------------------ \gererate text in table --------------------->

        # <------------------ compute height each cell --------------------->
        image_font = ImageFont.truetype(font=self.fonts[1], size=self.size)
        a_t = []
        for t in text:
            a_t += t
        _t = ' '.join(a_t)
        text_height = max([image_font.getsize(w)[1] for w in _t.split()])
        text_height += margin_top + margin_bottom
        self.heigh_each_cell = [text_height] * self.n_rows
        # <------------------ \compute height each cell --------------------->
        #
        for i, w in enumerate(self.width_each_cell[:-1]):
            self.cols.append(Col(width=w, margin_left=margin_left, margin_right=margin_right,
                                 index=i, line_left=Line(2, 1), line_right=Line(0, 0)))
        self.cols.append(Col(width=self.width_each_cell[-1], margin_left=margin_left, margin_right=margin_right,
                             index=len(self.width_each_cell), line_left=Line(2, 1), line_right=Line(2, 1)))

        for i, h in enumerate(self.heigh_each_cell[:-1]):
            self.rows.append(Row(height=h, margin_top=margin_top, margin_bottom=margin_bottom,
                                 index=i, line_top=Line(2, 1), line_bottom=Line(0, 0)))
        self.rows.append(Row(height=self.heigh_each_cell[-1], margin_top=margin_top, margin_bottom=margin_bottom,
                             index=len(self.heigh_each_cell), line_top=Line(2, 1), line_bottom=Line(2, 1)))

        self.table_height = sum([r.get_height() for r in self.rows])
        self.table_width = sum([c.get_width() for c in self.cols])

        for i, r in enumerate(self.rows):
            for j, c in enumerate(self.cols):
                if i == 0:
                    self.cells[i].append(
                        Cell(col=c, row=r, text=text[i][j], cell_id=j, font=self.fonts[random.choice([1, 3])],
                             align='left',
                             size=self.size))
                else:
                    self.cells[i].append(
                        Cell(col=c, row=r, text=text[i][j], cell_id=j, font=self.fonts[0], align='left',
                             size=self.size))

        left_or_right = 'right' if random.random() > 0.5 else 'center'
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if j != 0:
                    self.cells[i][j].align = left_or_right
        self.real_width = 0

    def get_col_start(self, index):
        return sum([c.get_width() for c in self.cols[:index]])

    def get_row_start(self, index):
        return sum([r.get_height() for r in self.rows[:index]])

    def get_table_bounding_box(self):
        return (
            sum([r.get_height() for r in self.rows]),
            min(sum([c.get_width() for c in self.cols[:-1]]), self.real_width))

    def draw(self, background_color=255):
        img = np.zeros((self.table_height, self.table_width, 3), dtype=np.uint8)
        img[:, :] = background_color

        # img = self.show_cell_by_xy(img, i, j, left=True, right=True, top=True, bottom=True)
        # underline for the first row ?
        underline = random.random() > 0.5
        widths = []
        for i_r, row in enumerate(self.rows):
            for i_c, col in enumerate(self.cols):
                if i_r == 0 and underline:
                    img, width = self.draw_text_cell_by_xy(img, i_r, i_c, underline)
                else:
                    img, width = self.draw_text_cell_by_xy(img, i_r, i_c)
                widths.append(width)

        self.real_width = max(widths)
        return img

    def draw_line(self, line: Line, img, xmin, xmax, ymin, ymax, orient='vertical'):
        assert orient in ['vertical', 'horizontal']
        if line.type == 0:
            return img
        line_thichness = line.line_thichness // (2 * line.type - 1)
        for i in range(2 * line.type - 1):
            if i % 2 == 0:
                if orient == 'horizontal':
                    start = xmin + i * line_thichness
                    img[start:start + line_thichness, ymin:ymax, :] = 0
                else:
                    start = ymin + i * line_thichness
                    img[xmin:xmax, start:start + line_thichness, :] = 0
        return img

    def show_cell_by_xy(self, img, x: int, y: int, left=False, right=False, top=False, bottom=False,
                        margin_left=False, margin_right=False, margin_top=False, margin_bottom=False):
        xmin = self.get_row_start(x) if not margin_top else self.get_row_start(x) + self.rows[x].margin_top
        xmax = xmin + self.rows[x].get_height() if not margin_bottom else xmin + self.rows[x].get_height() \
                                                                          - self.rows[x].margin_bottom
        ymin = self.get_col_start(y) if not margin_left else self.get_col_start(y) + self.cols[y].margin_left
        ymax = ymin + self.cols[y].get_width() if not margin_right else self.get_col_start(y) + self.cols[y].get_width() \
                                                                        - self.cols[y].margin_right

        if left:
            img = self.draw_line(line=self.cols[y].line_left, img=img, orient="vertical",
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymin + self.cols[y].line_left.line_thichness)
        if right:
            img = self.draw_line(line=self.cols[y].line_right, img=img, orient="vertical",
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymax - self.cols[y].line_right.line_thichness,
                                 ymax=ymax)
        if top:
            img = self.draw_line(line=self.rows[x].line_top, img=img, orient="horizontal",
                                 xmin=xmin,
                                 xmax=xmin + self.rows[x].line_top.line_thichness,
                                 ymin=ymin,
                                 ymax=ymax)
        if bottom:
            img = self.draw_line(line=self.rows[x].line_bottom, img=img, orient="horizontal",
                                 xmin=xmax - self.rows[x].line_bottom.line_thichness,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymax)
        return img

    def draw_text_cell_by_xy(self, img, x: int, y: int, underline=False):
        xmin, ymin, xmax, ymax = self.get_bounding_box_by_xy(x, y)
        img, width = draw_text(text=self.cells[x][y].text, font=self.cells[x][y].font, align=self.cells[x][y].align,
                               size=self.cells[x][y].size,
                               xmin=ymin,
                               ymin=xmin,
                               xmax=ymax, ymax=xmax, img=img, text_color='#000000,#282828', underline=underline)
        return img, width + self.cols[y].margin_right + self.cols[y].line_left.line_thichness

    def get_bounding_box_by_xy(self, x, y):
        # row
        xmin = self.get_row_start(x) + self.rows[x].line_top.line_thichness + self.rows[x].margin_top
        xmax = self.get_row_start(x) + self.rows[x].get_height() \
               - self.rows[x].line_bottom.line_thichness - self.rows[x].margin_bottom
        # col
        ymin = self.get_col_start(y) + self.cols[y].line_left.line_thichness + self.cols[y].margin_left
        ymax = self.get_col_start(y) + self.cols[y].get_width() \
               - self.cols[y].line_right.line_thichness - self.cols[y].margin_right
        return xmin, ymin, xmax, ymax


def draw_text(text, font, align, size, xmin, ymin, xmax, ymax, img, text_color='#000000,#282828', underline=False):
    assert align in ['left', 'right', 'center', 'justify']
    if type(img) == np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # BGR

    image_font = ImageFont.truetype(font=font, size=size)
    words = text.split(" ")
    space_width = image_font.getsize(" ")[0] * 1
    space_height = image_font.getsize(" ")[1] * 1
    words_width = [image_font.getsize(w)[0] for w in words]
    text_width = sum(words_width) + int(space_width) * (len(words) - 1)
    if xmax - text_width < xmin:
        return draw_text(' '.join(words[:-1]), font, align, size, xmin, ymin, xmax, ymax, img,
                         text_color='#000000,#282828', underline=underline)
    text_height = max([image_font.getsize(w)[1] for w in words])

    txt_draw = ImageDraw.Draw(img)
    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        random.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        random.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        random.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )
    if align == 'right':
        xmin = xmax - text_width
    elif align == 'center':
        xmin += (xmax - xmin - text_width) // 2

    for i, w in enumerate(words):
        start_x = sum(words_width[0:i]) + i * int(space_width) + xmin
        start_y = 0 + ymin
        txt_draw.text(
            (start_x, start_y),
            w,
            fill=fill,
            font=image_font,
        )
    if underline:
        underline = Image.new('RGB', (text_width, 2), (0, 0, 0))
        img.paste(underline, (xmin, ymin + text_height))

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    width = 0 if text == '' else xmin + text_width
    return img, width

def generate_date_data():
    ngay = random.randint(1, 31)
    thang = random.randint(1, 12)
    nam = random.randint(2000, 2050)
    return f"{ngay}/{thang}/{nam}"


def generate_tien(n):
    m = [str(random.randint(0, 9)) for _ in range(n)]
    m_ = [''.join(m[i:i + 3]) for i in range(0, n, 3)]
    s = '.'.join(m_)[::-1]
    return s


if "PycharmProjects" in os.getcwd():
    # may cong ty
    filename = "/home/andn/PycharmProjects/TextRecognitionDataGenerator/texts/VNESEcorpus_5.txt"
else:
    # server 123.30.171.216
    filename = "/data/andn/TextRecognitionDataGenerator/texts/VNESEcorpus_5.txt"
with open(filename) as f:
    lines = f.readlines()

assert len(lines) > 0


def generate_string(n):
    line = random.choice(lines)
    words = line.split()
    if len(words) < n:
        return line
    start = random.randint(0, len(words) - n)
    return ' '.join(words[start:start + n])


def generate_rate():
    return f'{random.randint(0, 100)}%'


def map_pixel_to_size(pixel):
    a = {11: 10, 12: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 20: 17, 21: 18, 21: 19, 22: 20, 23: 21, 25: 22,
         26: 23, 27: 24, 28: 25, 30: 26, 31: 27, 31: 28, 32: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 40: 35, 41: 36,
         41: 37, 42: 38, 43: 39}

    while pixel not in a:
        pixel += 1

    return a[pixel]


if __name__ == '__main__':
    while True:
        Table = random.choice([Table1, Table2])
        show_img(Table().draw())
# img = np.zeros((500, 500, 3), dtype=np.uint8) + 255
# fonts = ['font/times.ttf', 'font/timesbd.ttf', 'font/timesi.ttf', 'font/timesbi.ttf']
# draw_text(text="Đào Ngọc An", font=fonts[0], align='center', size=48, xmin=200, ymin=200, xmax=500, ymax=500, img=img,
#           text_color='#000000,#282828').show()

# chữ viết thường, chữ nghiêng, chữ viết đậm, chữ viết đậm nghiêng
#
