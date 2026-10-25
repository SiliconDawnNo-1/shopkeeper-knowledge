import base64
import os
import re
import sys
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from common.config.lm_config import lm_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_running_task, add_done_task


# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    # return Path(filename).suffix.lower() in IMAGE_EXTENSIONS
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state)->tuple[str, Path,Path]:
    # 从状态中获取获取md_path
    md_path = state.get("md_path")

    # 判断状态中是否存在md_path
    if not md_path:
        logger.error("在状态中没有找到md_path,流程终止")
        raise ValueError("在状态中没有找到md_path,流程终止")

    # 将md_path封装为一个Path实例
    md_path_obj: Path = Path(md_path)

    # 判断文件是否存在
    if not md_path_obj.is_file():
        logger.error(f"{md_path_obj}不存在或者不是一个文件")
        raise FileNotFoundError(f"{md_path_obj}不存在或者不是一个文件")

    # 获取md文件内容
    md_content = md_path_obj.read_text(encoding="utf-8")

    # 获取md文件中图片存放路径
    md_images_dir_obj: Path = md_path_obj.parent / "images"

    return md_content, md_path_obj, md_images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content, md_images_dir_obj)->list[tuple[str,str,tuple[str, str]]]:
    """
    获取图片信息
    :param md_content:          md内容
    :param md_images_dir_obj:   md图片所在的目录
    :return:                    图片的信息list[tuple[图片名,图片路径,tuple[上文, 下文]]]
    """
    image_info_list:list[tuple[str,str,tuple[str, str]]] = []

    # 对images目录下的内容进行遍历
    for md_file_obj in md_images_dir_obj.iterdir():
        # 获取当前图片的名字
        image_name = md_file_obj.name
        # 获取当前图片的路径
        image_path = str(md_file_obj)

        # 过滤掉不支持图片
        if not is_supported_image(image_name):
            logger.warning(f"跳过非图片文件：{image_name}")
            continue

        # 定义正则匹配
        rep = re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+r".*?\)")
        search_match = rep.search(md_content)

        if not search_match:
            # 为空,真没有匹配到
            logger.warning(f"{image_name}没有被md_content引用引用,跳过,直接下一次!!")
            continue

        # 如果程序能够走到这里，说明当前遍历出来的图片在md_content找到了
        # 获取md_content中匹配的图片的起始位置
        start = search_match.start()
        end = search_match.end()

        # 获取图片的上文
        pre_content = md_content[max(0,start-100):start]
        # 获取图片的下文
        post_content = md_content[end: min(end + 100,len(md_content))]
        logger.debug( f"{image_name}在md_content被引用,引用的位置:{start}:{end},截取的上文:{pre_content} , 下文:{post_content}")
        image_info_list.append((image_name,image_path,(pre_content,post_content)))

    logger.info(f"所有图片的上下文信息已经识别完毕,数量为:{len(image_info_list)}")
    return image_info_list


@step_log("step_3_image_summary")
def step_3_image_summary(image_info_list, root_folder):
    """
    调用视觉模型   获取图片摘要
    :param image_info_list:     图片信息 list[(图片名,图片路径,(上文,下文))]
    :param root_folder:         md文件名
    :return:                    dict{key:图片名,value:摘要}
    """
    image_summary_dict = {}

    # 获取视觉模型客户端对象
    vl_model = get_llm_client(lm_config.vl_model)
    # 对图片信息列表进行遍历
    for image_name,image_path,image_content in image_info_list:
        # 拼接提示词
        prompt_text = load_prompt("image_summary",root_folder=root_folder,image_content= image_content)

        # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        # image_data = base64.b64encode(requests.get(image_url).content).decode("utf-8")
        # base64：  将图片转换为字符串(方便进行传递)      以及将字符串转换为图片
        image_path_obj = Path(image_path)
        image_data = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        # 封装消息对象
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_data}"},
                },
            ]
        )
        # 封装调用链
        chains = vl_model | StrOutputParser()

        # 在调用模型之前，使用滑动窗口限速器进行限速
        apply_api_rate_limit()

        # 调用链获取结果
        image_summary = chains.invoke([message])

        image_summary_dict[image_name] = image_summary
        logger.debug(f"完成:{image_name}的视觉识别,对应的含义:{image_summary}")

    return image_summary_dict


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    """
    # TODO 1. 添加节点到运行时列表
    add_running_task(state.get("task_id"),"node_md_img")
    # TODO 2. 从状态中获取数据并进行校验
    md_content, md_path_obj, md_images_dir_obj = step_1_validate_and_get_data(state)

    # 判断md文件中是否存在图片(因为有些pdf中可能没有图片--那么解析后的md文件中就没有图片，images目录就不存在)
    if (not md_images_dir_obj) or (len(list(md_images_dir_obj.iterdir()))==0):
        logger.info(f"{md_path_obj}对应的md,没有图片内容,无需后续处理,直接跳出!!")
        add_done_task(state.get("task_id"), "node_md_img")
        return state

    # TODO 3. 获取图片信息和图片上下文   ----- list[tuple(图片名,图片路径,tuple(上文,下文))]
    image_info_list: list[tuple[str, str, tuple[str, str]]] = step_2_scan_images(md_content, md_images_dir_obj)

    # TODO 4. 调用视觉模型获取图片的摘要  ---- dict{key:图片名,value:图片摘要}
    image_summary_dict: dict[str, str] = step_3_image_summary(image_info_list, md_path_obj.stem)

    # TODO 5. 将图片上传到minio服务器    ---- dict{key:图片名,value:图片的网络地址}
    # TODO 6. 替换md文件中的内容，将摘要以及网络地址替换到md文件中
    # TODO 7. 将新的md文件进行磁盘的持久化备份
    # TODO 8. 更新状态   md_content
    # TODO 9. 添加节点到已完成时列表
    add_done_task(state.get("task_id"),"node_md_img")
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""

    test_md_path = r"D:\dev\workspace\shopkeeper-brain-0810\output\hak180产品安全手册\hak180产品安全手册.md"

    # 构造测试状态对象，模拟流程入参
    test_state = {
        "md_path": test_md_path,
        "task_id": "test_task_123456"
    }
    # 执行核心处理流程
    result_state = node_md_img(test_state)
    logger.info(f"本地测试完成 - 处理结果状态：{result_state}")