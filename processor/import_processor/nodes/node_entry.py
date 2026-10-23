import sys
from pathlib import Path

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.task_utils import add_running_task, add_done_task


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    """
    # TODO 1. 添加节点到运行时列表
    add_running_task(state.get("task_id"),"node_entry")

    # TODO 2. 从状态中获取本地文件路径
    local_file_path = state.get("local_file_path")

    if not local_file_path:
        logger.warning("在状态中没有找到local_file_path，流程终止")
        return state

    # TODO 3. 根据文件后缀判断文件类型   并根据类型对状态中的属性进行更新
    if local_file_path.endswith(".md"):
        # 当前处理的是md文件
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False
        state["md_path"] = local_file_path
        state["pdf_path"] = None
    elif local_file_path.endswith(".pdf"):
        # 当前处理的是pdf文件
        state["is_md_read_enabled"] = False
        state["is_pdf_read_enabled"] = True
        state["md_path"] = None
        state["pdf_path"] = local_file_path
    else:
        # 不支持的文件类型
        logger.warning("不支持的文件类型")
        return state

    # TODO 4. 提取文件名  更新状态属性
    local_file_path_obj = Path(local_file_path)
    # .stem 获取文件名 不含后缀   .parent 获取文件父级目录  .name 获取文件名 带后缀  .
    file_title = local_file_path_obj.stem
    state["file_title"] = file_title

    # TODO 5. 添加节点到已完成列表
    add_done_task(state.get("task_id"),"node_entry")

    return state


if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    # test_state1 = create_default_state(
    #     task_id="test_task_001",
    #     local_file_path="联想海豚用户手册.md"
    # )
    # print(node_entry(test_state1))

    # 测试2: MD文件
    # test_state2 = create_default_state(
    #     task_id="test_task_002",
    #     local_file_path="小米用户手册.docx"
    # )
    # print(node_entry(test_state2))

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    print(node_entry(test_state3))

    logger.info("===== 结束node_entry节点单元测试 =====")