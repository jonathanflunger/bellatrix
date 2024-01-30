import os

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def file_save(fig, dir_out, file_name):
    fig.tight_layout()
    fig.set_size_inches((16/9)*5, 5)
    fig.savefig(dir_out + file_name + ".pdf", transparent=True, dpi = 200)

def save_plot(fig, file_name):
    dir_out =  "../plots/"
    create_dir(dir_out)
    file_save(fig, dir_out, file_name)