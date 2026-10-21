from typing import TypedDict
import copy
from common.logging.logger import logger

class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点产生和消费的数据字段。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如 state["task_id"]、state.get("chunks")）。
    """
    # 任务id，每一次上传都会生成不重复的数据
    task_id: str

    # --- 流程控制标记 ---
    # 判断是否是md文件的标记
    is_md_read_enabled: bool
    # 判断是否是pdf文件的标记
    is_pdf_read_enabled: bool

    # --- 路径相关 ---
    # 文件夹地址  (pdf -> md  -> 输出的文件夹地址)
    local_dir: str
    # 传入文件地址 不确定md pdf
    local_file_path: str
    # 文件名
    file_title: str
    # pdf地址 文件 <- local_file_path
    pdf_path: str
    # md地址 文件 <- local_file_path
    md_path: str

    # --- 内容数据 ---
    # md文件中的内容
    md_content: str
    # 经过文档切分后  将一个个chunk放到chunks中
    chunks: list
    # 主体名(当前文档描述的商品名)
    item_name: str

    # --- 数据库相关 ---
    #经过嵌入操作后的chunk
    embeddings_content: list


graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖。
    """
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(graph_default_state)

if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    logger.info(state)