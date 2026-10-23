import os
import shutil
import sys
import time
from pathlib import Path

import requests

from common.config.mineru_config import mineru_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import add_running_task, add_done_task


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):

    # 从状态中获取pdf_path
    pdf_path = state.get("pdf_path")
    # 从状态中获取local_dir
    local_dir = state.get("local_dir")

    if not pdf_path:
        logger.error("在状态中没有找到pdf_path,流程终止")
        raise ValueError("在状态中没有找到pdf_path,流程终止")

    if not local_dir:
        logger.warning("在状态中没有找到local_dir，手动指定")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = local_dir

    # 将pdf_path由字符串封装为Path对象
    pdf_path_obj = Path(pdf_path)
    # 这步可以不做 ，因为PROJECT_ROOT就是Path类型，拼接后结果还是Path
    local_dir_obj = Path(local_dir)

    # 判断pdf文件是否存在   is_file:判断是否存在、判断是不是文件
    if not pdf_path_obj.is_file():
        logger.error(f"{pdf_path_obj}不存在或者不是一个文件，流程终止")
        raise RuntimeError(f"{pdf_path_obj}不存在或者不是一个文件，流程终止")

    if not local_dir_obj.is_dir():
        logger.warning(f"{local_dir_obj}不存在或者不是一个目录，手动创建")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj


@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj):
    """向minerU发送请求 ，minerU解析pdf，返回zip地址"""
    # ===配置校验===
    if (not mineru_config.api_key) or (not mineru_config.base_url):
        logger.error(f"minerU配置错误,请检查minerU配置!")
        raise ValueError("minerU配置错误,请检查minerU配置!")

    # ===申请上传地址===
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": pdf_path_obj.name, "data_id": pdf_path_obj.stem}
        ],
        "model_version": "vlm"
    }

    # 向服务器发送请求获取上传地址
    http_response = requests.post(url, headers=header, json=data)
    # 获取请求响应的状态码
    http_status_code = http_response.status_code
    if http_status_code != 200:
        logger.error(f"申请上传地址失败,返回状态码为:{http_status_code},请检查minerU配置!")
        raise RuntimeError(f"申请上传地址失败,返回状态码为:{http_status_code},请检查minerU配置!")

    # 如果服务器响应成功，获取响应信息
    # http_response.text        以文本字符串形式获取响应数据
    # http_response.content     获取响应的二进制数据
    # http_response.json()      将响应的信息转换为json格式-->python中的dict
    http_response_dict = http_response.json()

    http_response_dict_code = http_response_dict.get("code")
    http_response_dict_msg = http_response_dict.get("msg")
    if http_response_dict_code != 0:
        logger.error(f"申请地址网络状态成功!但是业务失败!错误码:{http_response_dict_code},失败信息:{http_response_dict_msg}")
        raise RuntimeError(f"申请地址网络状态成功!但是业务失败!错误码:{http_response_dict_code},失败信息:{http_response_dict_msg}")

    batch_id = http_response_dict.get("data").get("batch_id")
    file_upload_urls = http_response_dict['data']['file_urls']

    if not file_upload_urls:
        logger.error(f"没有申请到上传地址")
        raise RuntimeError(f"没有申请到上传地址")
    if not batch_id:
        logger.error(f"没有批量操作标识")
        raise RuntimeError(f"没有批量操作标识")

    file_upload_url = file_upload_urls[0]
    logger.info(f">>>申请地址成功{file_upload_url}>>>")

    # ===上传文件===
    pdf_file_data = pdf_path_obj.read_bytes()
    # 注意：如果用以下方式，有一定的概率会报错
    # put 使用的是预签名机制，如果客户端开启了代理，在向存储服务器发送请求的时候，在请求头中会携带额外信息，导致校验失败
    # upload_response = requests.put(file_upload_url, data=pdf_file_data)
    # 解决方案：使用session发送请求，让trust_env= False   就算是开启了代理，也会发送纯净版的消息
    with requests.session() as session:
        session.trust_env = False
        # 上传文件
        upload_response =  session.put(url=file_upload_url, data=pdf_file_data)
        if upload_response.status_code != 200:
            logger.error(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
            raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")

        logger.info(f">>>向{file_upload_url}上传{pdf_path_obj}成功>>>")

    # ===轮询发送请求获取mineru的解析结果(zip的路径)===
    # 轮询就是死循环,终止条件: 1. 拿到结果  2. 失败了 3. 超时
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"

    # 设置超时时间  单位秒
    timeout = 600
    # 设置轮询间隔时间
    interval_time = 3
    # 请求的起始时间
    start_time = time.time()

    while True:
        # 判断是否超时
        if time.time() - start_time > timeout:
            logger.error(f"轮询超时,请检查minerU配置!")
            raise TimeoutError(f"轮询超时,请检查minerU配置!")

        try:
            poll_response = requests.get(poll_url, headers=header)
        except Exception as e:
            logger.warning(f"请求出现异常!可以稍后重试!!")
            time.sleep(interval_time)
            continue

        # 获取响应的状态码
        poll_response_status_code = poll_response.status_code
        if poll_response_status_code != 200:
            # 如果是服务器出问题，可以尝试重试
            if 500 <= poll_response_status_code < 600:
                logger.warning(f"可有修复的网络异常,状态码为:{poll_response_status_code}")
                time.sleep(interval_time)
                continue
            else:
                logger.error(f"不可修复的网络状态异常,状态码为:{poll_response_status_code}")
                raise RuntimeError(f"不可修复的网络状态异常,状态码为:{poll_response_status_code}")

        # 如果服务器响应成功，获取响应的数据
        poll_response_dict = poll_response.json()
        poll_response_dict_code = poll_response_dict.get("code")
        poll_response_dict_msg = poll_response_dict.get("msg")
        # 如果服务器响应成功，但是业务接口状态码 != 0,说明业务接口调用失败
        if poll_response_dict_code != 0:
            logger.error(f"轮询业务异常,错误码:{poll_response_dict_code},失败信息:{poll_response_dict_msg}")
            raise RuntimeError(f"轮询业务异常,错误码:{poll_response_dict_code},失败信息:{poll_response_dict_msg}")


        # 获取解析的结果
        extract_result = poll_response_dict.get("data").get("extract_result")[0]
        extract_result_state = extract_result.get("state")
        if extract_result_state == "done":
            # 服务器解析成功
            zip_url = extract_result.get("full_zip_url")
            if not zip_url:
                logger.error(f"已经完成了解析,但是zip地址为空!!")
                raise RuntimeError(f"已经完成了解析,但是zip地址为空!!")
                # 5. 返回压缩地址即可
            logger.info(f">>>获取mineru服务器的解析结果：{zip_url}>>>")
            return zip_url
        elif extract_result_state == "failed":
            logger.error(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
        else:
            logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
            time.sleep(interval_time)
            continue


@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url, local_dir_obj, stem):
    # ===发送请求 下载解析结果===
    response = requests.get(zip_url,timeout=60)

    if response.status_code != 200:
        logger.error(f"从{zip_url}下载文件失败!")
        raise RuntimeError(f"从{zip_url}下载文件失败!")

    # 指定zip压缩包存放的路径
    md_zip_path_obj:Path = local_dir_obj / f"{stem}_result.zip"

    # 将压缩包内容写到指定的路径对象中
    md_zip_path_obj.write_bytes(response.content)

    # ===解压===
    # 指定解压目录
    extract_path_obj:Path = local_dir_obj / stem

    # 为了避免路径已经存在，对新解析的内容造成污染，先将解压目录删除掉
    if extract_path_obj.is_dir():
        shutil.rmtree(extract_path_obj)

    # 删完之后，重新创建解压的目标目录
    extract_path_obj.mkdir(parents=True, exist_ok=True)

    # 解压
    shutil.unpack_archive(md_zip_path_obj,extract_path_obj)

    #===查找解压后的目录下的md文件===
    # rglob:以递归的方式查找和pattern匹配的文件
    # glob: 查找和pattern匹配的文件
    # iterdir: 列出目录对象下的所有文件
    md_file_list = list(extract_path_obj.rglob("*.md"))

    # 优先级1：找和被解析的pdf同名的md
    for md_file_obj in md_file_list:
        if md_file_obj.stem == stem:
            logger.info(f"文件{md_file_obj}解压成功")
            return md_file_obj
    # 优先级2：找full.md
    target_file_obj = None
    for md_file_obj in md_file_list:
        if md_file_obj.name.lower() == "full.md":
            target_file_obj = md_file_obj
            logger.info(f"文件解压成功,但是名字是full.md,后续需要重命名")
            break
    # 优先级3：从列表中找第一个就行
    if target_file_obj is None:
        target_file_obj = md_file_list[0]


    # 如果没有找到同名的md文件，需要对找到的md文件进行改名
    # rename底层原理：先删除，再创建   with_name：只是修改文件名  创建新的文件对象，不会对磁盘进行操作
    final_md_path_obj = target_file_obj.rename(target_file_obj.with_name(f"{stem}.md"))
    logger.info(f"文件{final_md_path_obj}解压成功")
    return final_md_path_obj


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    """
    # TODO 1. 添加节点到运行时列表
    add_running_task(state.get("task_id"),"node_pdf_to_md")
    # TODO 2. 从状态中获取数据并进行校验
    pdf_path_obj, local_dir_obj = step_1_validate_and_get_data(state)
    # TODO 3. 调用MinerU进行解析并获取结果
    zip_url:str = step_2_upload_and_poll(pdf_path_obj)
    # TODO 4. 下载结果并进行解压
    md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
    # TODO 5. 更新状态 - md_path
    state["md_path"] = str(md_path)
    # TODO 6. 添加节点到已完成列表
    add_done_task(state.get("task_id"),"node_pdf_to_md")
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    # test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    # test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)
    test_pdf_path = PROJECT_ROOT / "doc" / "hak180产品安全手册.pdf"

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path= str(test_pdf_path),
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    res_state = node_pdf_to_md(test_state)

    logger.success(res_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")