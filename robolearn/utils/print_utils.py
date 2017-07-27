import sys


class PrintColors:
    def __init__(self):
        self.colors = {
        'RED': "\033[1;31m",
        'GREEN': "\033[1;32m",
        'YELLOW': "\033[1;33m",
        'BLUE': "\033[1;34m",
        'MAGENTA': "\033[1;35m",
        'CYAN': "\033[1;36m",
        'GRAY': "\033[1;37m",
        'RESET': "\033[0;0m",
        'BOLD': "\033[;1m",
        'REVERSE': "\033[;7m",
        }

    def change(self, color):
        if color.upper() not in self.colors.keys():
            raise ValueError("Wrong color!!")
        sys.stdout.write(self.colors[color.upper()])

    def reset(self):
        sys.stdout.write(self.colors['RESET'])

change_print_color = PrintColors()

def print_skull():
    #print("   _                   _ ")
    #print(" _( )                 ( )_ ")
    #print("(_, |      __ __      | ,_) ")
    #print("   \'\    /  ^  \    /'/ ")
    #print("    '\'\,/\      \,/'/' ")
    #print("      '\| []   [] |/' ")
    #print("        (_  /^\  _) ")
    #print("          \  ~  / ")
    #print("          /HHHHH\ ")
    #print("        /'/{^^^}\'\ ")
    #print("    _,/'/'  ^^^  '\'\,_ ")
    #print("   (_, |           | ,_) ")
    #print("     (_)           (_) ")

    print("  {}          {} ")
    print("   \  _---_  / ")
    print("    \/     \/ ")
    print("     |() ()| ")
    print("      \ + / ")
    print("     / HHH  \ ")
    print("    /  \_/   \ ")
    print("  {}          {} ")


    # print('         .e$$$$e.  ')
    # print('       e$$$$$$$$$$e  ')
    # print('      $$$$$$$$$$$$$$  ')
    # print('     d$$$$$$$$$$$$$$b  ')
    # print('     $$$$$$$$$$$$$$$$  ')
    # print('    4$$$$$$$$$$$$$$$$F  ')
    # print('    4$$$$$$$$$$$$$$$$F  ')
    # print('     $$$" "$$$$" "$$$  ')
    # print('     $$F   4$$F   4$$  ')
    # print('     "$F   4$$F   4$"  ')
    # print('      $$   $$$$   $P  ')
    # print('      4$$$$$"^$$$$$%  ')
    # print('       $$$$F  4$$$$  ')
    # print('        "$$$ee$$$"  ')
    # print('        . *$$$$F4  ')
    # print('         $     .$  ')
    # print('         "$$$$$$"  ')
    # print('          ^$$$$  ')
    # print(' 4$$c       ""       .$$r  ')
    # print(' ^$$$b              e$$$"  ')
    # print(' d$$$$$e          z$$$$$b  ')
    # print('4$$$*$$$$$c    .$$$$$*$$$r  ')
    # print(' ""    ^*$$$be$$$*"    ^"  ')
    # print('          "$$$$"  ')
    # print('        .d$$P$$$b  ')
    # print('       d$$P   ^$$$b  ')
    # print('   .ed$$$"      "$$$be.  ')
    # print(' $$$$$$P          *$$$$$$  ')
    # print('4$$$$$P            $$$$$$"  ')
    # print(' "*$$$"            ^$$P  ')
    # print('    ""              ^"  ')


def print_robotio():
    print("       @@@@@@@@@@@@@  ")
    print("     @%@@//@@@@@//@@/@  ")
    print("     @%@@///@@@@//@@/@  ")
    print("     @%@@@@@@@@@@@@@/@  ")
    print("     @%@@,,@,,,@,,@@/@  ")
    print("       @@,,@,,,@,,@@  ")
    print("       @@@@@@@@@@@@@  ")
    print("           #####  ")
    print("  @@@%@@@@@@@@@@@ @ @@@@@,  ")
    print("  @@@%####@...&@ @%@ @@@@,  ")
    print("  @@@%@@@@@@@@@@@@(@@@@@@,  ")
    print("  @@@%@#############@@@@@,  ")
    print("  @@@%@@@@@@@@@@@@@@@@@@@,  ")
    print("  @@@    @@@@@@@@@    @@@,  ")
    print("  @@@  @@@@@@@@@@@@@  @@@,  ")
    print(" @@@@@ @@@@@   @@@@@ #@@@@  ")
    print(" @@ @@ @@@@@   @@@@@ #@ @@  ")
    print("       @@@@@   @@@@@  ")
    print("       @@@@@   @@@@@  ")
    print("       @@@@@   @@@@@  ")
    print("       @@@@@   @@@@@  ")
    print("       @@@@@   @@@@@  ")
    print("       @@@@@   @@@@@  ")
    print("       @@@@@   @@@@@  ")
    print("       @@@@@   @@@@@  ")
    print("      %%%%%%% %%%%%%%  ")
    print("      &&&&&&& &&&&&&&  ")


def print_robotio_big():
               print("           @@@@@@@@@@@@@@@@@@@@@@@@@@@            ")
               print("           @@@@@@@@@@@@@@@@@@@@@@@@@@@            ")
               print("       ....@@@@@@%&@@@@@@@@@@@@@@@@@@@ ...        ")
               print("       @@@%@@@@//////@@@@@@@@/////&@@@/@@@        ")
               print("       @@@%@@@&//////@@@@@@@@/////#@@@/@@@        ")
               print("       @@@%@@@@@///#@@@@@@@@@@@(%@@@@@/@@@        ")
               print("       @@@%@@@@@@@@@@@@@@@@@@@@@@@@@@@/@@@        ")
               print("       @@@%@@@@,,,,,@@,,,,,@@,,,,,@@@@/@@@        ")
               print("       @@@%@@@@,,,,,@@,,,,,@@,,,,,@@@@/@@@        ")
               print("       @@@%@@@@,,,,,@@,,,,,@@,,,,,@@@@/@@@        ")
               print("       @@@%@@@@,,,,,@@,,,,,@@,,,,,@@@@/@@@        ")
               print("           @@@@,,,,,@@,,,,,@@,,,,,@@@@            ")
               print("           @@@@@@@@@@@@@@@@@@@@@@@@@@@            ")
               print("                   #@@@@@@@@@                     ")
               print("                   #@@@@@@@@@                     ")
               print(" @@@@@@@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@, ")
               print(" @@@@@@@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@, ")
               print(" @@@@@@@%@@@@@@@@@@@@@@@@@@@@@@ @@@@@ @@@@@@@@@@, ")
               print(" @@@@@@@%@#######@@.......&@@@ @@%%%@@ @@@@@@@@@, ")
               print(" @@@@@@@%@#######@@.......&@@@,@@@%@@@,@@@@@@@@@, ")
               print(" @@@@@@@%@@@@@@@@@@@@@@@@@@@@@@& @@@ &@@@@@@@@@@, ")
               print(" @@@@@@@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@, ")
               print(" @@@@@@@%@@###########################@@@@@@@@@@, ")
               print(" @@@@@@@%@@###########################@@@@@@@@@@, ")
               print(" @@@@@@@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@, ")
               print(" #######       ,,,,,,,,,,,,,,,,,,,       #######. ")
               print(" @@@@@@@       %@@@@@@@@@@@@@@@@@@       @@@@@@@, ")
               print(" @@@@@@@       %@@@@@@@@@@@@@@@@@@       @@@@@@@, ")
               print(" @@@@@@@   .@@@@@@@@@@@@@@@@@@@@@@@@@/   @@@@@@@, ")
               print(" @@@@@@@   .@@@@@@@@@@@@@@@@@@@@@@@@@/   @@@@@@@, ")
               print("@@@@@@@@@   *************************,  #@@@@@@@@ ")
               print("@@@@@@@@@  .@@@@@@@@@       @@@@@@@@@/  #@@@@@@@@ ")
               print("@@@   @@@  .@@@@@@@@@       @@@@@@@@@/  #@@   @@@ ")
               print("@@@   @@@  .@@@@@@@@@       @@@@@@@@@/  #@@   @@@ ")
               print("&&&   &&&  .@@@@@@@@@       @@@@@@@@@/  #&&   &&& ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .%%%%%%%%%       %%%%%%%%%*            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("           .@@@@@@@@@       @@@@@@@@@/            ")
               print("         /@@@@@@@@@@@@@/ .@@@@@@@@@@@@@%          ")
               print("         /@@@@@@@@@@@@@/ .@@@@@@@@@@@@@%          ")

