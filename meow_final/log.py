import os
from datetime import datetime
from inspect import currentframe, getframeinfo


class MeowLogger(object):
    """项目内的简易日志工具，用统一格式输出运行信息。"""

    def __init__(self):
        """默认输出到控制台；调用 setLogFile 后可以改为写入文件。"""
        self.logf = None

    def __del__(self):
        """对象销毁时关闭已打开的日志文件。"""
        if self.logf is not None:
            self.logf.close()

    def __header(self, pid):
        """生成日志前缀，包含时间、文件名、行号，必要时包含进程号。"""
        now = datetime.now()
        frameInfo = getframeinfo(currentframe().f_back.f_back)
        if pid:
            return "[\033[90m{}|\033[0m{}:{}|{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno, os.getpid())
        return "[\033[90m{}|\033[0m{}:{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno)

    def setLogFile(self, filename):
        """设置日志文件；后续日志会写入该文件而不是打印到屏幕。"""
        if self.logf is not None:
            self.logf.close()
        self.logf = open(filename, "w")

    def log(self, content, muted=False):
        """底层输出函数，支持静默模式、文件输出和控制台输出。"""
        if muted:
            return
        if self.logf is not None:
            self.logf.write(content + "\n")
            self.logf.flush()
            return
        print(content)

    def inf(self, line, pid=False, muted=False):
        """输出普通信息日志。"""
        self.log(self.__header(pid) + line, muted)

    def grey(self, line, pid=False, muted=False):
        """输出灰色日志，常用于不太重要的调试信息。"""
        self.log("{}\033[90m{}\033[0m".format(self.__header(pid), line), muted)

    def red(self, line, pid=False, muted=False):
        """输出红色日志，常用于错误或严重警告。"""
        self.log("{}\033[91m{}\033[0m".format(self.__header(pid), line), muted)

    def green(self, line, pid=False, muted=False):
        """输出绿色日志，常用于成功提示。"""
        self.log("{}\033[92m{}\033[0m".format(self.__header(pid), line), muted)

    def yellow(self, line, pid=False, muted=False):
        """输出黄色日志，常用于可恢复的警告。"""
        self.log("{}\033[93m{}\033[0m".format(self.__header(pid), line), muted)

    def blue(self, line, pid=False, muted=False):
        """输出蓝色日志，常用于强调某个阶段。"""
        self.log("{}\033[94m{}\033[0m".format(self.__header(pid), line), muted)

    def pink(self, line, pid=False, muted=False):
        """输出粉色日志，供需要区分的提示使用。"""
        self.log("{}\033[95m{}\033[0m".format(self.__header(pid), line), muted)

    def cyan(self, line, pid=False, muted=False):
        """输出青色日志，供需要区分的提示使用。"""
        self.log("{}\033[96m{}\033[0m".format(self.__header(pid), line), muted)


log = MeowLogger()
