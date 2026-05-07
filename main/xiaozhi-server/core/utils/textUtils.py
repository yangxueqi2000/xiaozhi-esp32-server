import json
import re

TAG = __name__

EMOJI_MAP = {
    "😀": "happy",
    "😃": "happy",
    "😄": "happy",
    "😁": "funny",
    "😆": "funny",
    "😊": "loving",
    "😍": "loving",
    "😘": "kissy",
    "😎": "cool",
    "😢": "crying",
    "😭": "crying",
    "😠": "angry",
    "😡": "angry",
    "😮": "surprised",
    "😱": "shocked",
    "🤔": "thinking",
    "😴": "sleepy",
    "😜": "silly",
    "😕": "confused",
    "😐": "neutral",
    "😳": "embarrassed",
    "😉": "winking",
    "😋": "delicious",
    "😌": "relaxed",
    "😏": "confident",
    "😞": "sad",
}

EMOJI_RANGES = [
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
]

PUNCTUATION_SET = {
    ",",
    ".",
    "!",
    "?",
    ":",
    ";",
    "-",
    "~",
    "[",
    "]",
    "(",
    ")",
    "，",
    "。",
    "！",
    "？",
    "：",
    "；",
    "（",
    "）",
    "【",
    "】",
    "\"",
    "'",
    "“",
    "”",
    "‘",
    "’",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;\n])")

BACKSTAGE_PROCESS_KEYWORDS = (
    "读取",
    "记录",
    "写入",
    "补全",
    "校验",
    "切到下一步",
    "进入下一步",
    "推进到下一步",
    "读取当前步骤",
    "读取当前子步骤",
    "读取动作说明",
    "读取唯一动作",
    "确认当前步骤",
    "确认当前子步骤",
    "切换到当前步骤",
    "切换到当前子步骤",
    "建立会话",
    "会话",
    "工具",
    "调用",
    "后台",
    "检索",
    "加载",
    "恢复记录",
    "导出",
    "拍照确认",
    "强制拍照确认",
    "只给你当前动作",
    "只给你一个动作",
    "只告诉你当前要做的一个动作",
    "只告诉你这一句操作指令",
    "当前要做的一个动作",
    "动作说明",
    "操作指令",
    "下一句操作指令",
)

BACKSTAGE_PREFIXES = (
    "收到你的完成反馈",
    "收到",
    "继续",
    "我先",
    "我会先",
    "我现在",
    "我把",
    "我这边先",
    "我这边现在",
    "我继续",
    "我马上",
    "正在",
    "先帮你",
    "先给你",
    "接着",
    "然后",
    "马上",
    "这一步是",
    "这一步先是",
)

BACKSTAGE_LEADING_PATTERNS = [
    re.compile(r"^(?:当前要做的一个动作|给你当前要做的一个动作)[。！？；，、,\s]*"),
    re.compile(
        r"^这一步(?:先)?是[^。！？；]{0,120}"
        r"(?:总览确认|总览说明|总览阶段|概览确认)"
        r"[^。！？；]{0,160}"
        r"(?:记录什么|进入共同准备阶段|进入共同准备的第一步|开始共同准备阶段|带你进入)"
        r"[^。！？；]{0,120}[，、,\s]*"
    ),
    re.compile(
        r"^收到[^。！？；]{0,40}(?:完成反馈|反馈|结果)[^。！？；]{0,20}[，、,\s]*"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我先|我会先|我现在|我把|我这边先|我这边现在|我继续|我马上|正在|先帮你|先给你)"
        r"[^。！？；]{0,120}"
        r"(?:记录|记上|写上|写入|补全|校验|切到下一步|进入下一步|推进到下一步|推进到|带你进入|"
        r"进入共同准备阶段|进入共同准备的第一步|"
        r"读取当前步骤|读取当前子步骤|"
        r"读取动作说明|读取唯一动作|确认当前步骤|确认当前子步骤|切换到当前步骤|切换到当前子步骤|"
        r"建立会话|工具|调用|后台|检索|加载|恢复记录|导出|拍照确认|强制拍照确认|"
        r"只给你当前动作|只给你一个动作|只告诉你当前要做的一个动作|只告诉你这一句操作指令|"
        r"当前要做的一个动作|动作说明|操作指令|下一句操作指令)"
        r"[^。！？；]{0,120}[，、,\s]*"
    ),
    re.compile(
        r"^(?:接着|然后|马上)"
        r"[^。！？；]{0,120}"
        r"(?:切到下一步|进入下一步|推进到下一步|只给你当前动作|只给你一个动作|"
        r"只告诉你当前要做的一个动作|只告诉你这一句操作指令|当前要做的一个动作|"
        r"动作说明|操作指令|下一句操作指令)"
        r"[^。！？；]{0,120}[，、,\s]*"
    ),
    re.compile(
        r"^继续"
        r"[^。！？；]{0,120}"
        r"(?:记录|校验|确认下一步动作|当前步骤|当前子步骤|拍照确认|推进)"
        r"[^。！？；]{0,120}[，、,\s]*"
    ),
]

BACKSTAGE_FILLER_PATTERNS = [
    re.compile(r"^(?:继续|收到|好的|好)[。！？；，、,\s]*$"),
]

BACKSTAGE_REWRITE_PATTERNS = [
    (
        re.compile(
            r"^这一步(?:先)?是[^。！？；]{0,120}"
            r"(?:总览确认|总览说明|总览阶段|概览确认)"
            r"[^。！？；]{0,160}[。！？；，、,\s]*$"
        ),
        "先确认整体安排，准备好后就开始共同准备阶段。",
    ),
    (
        re.compile(
            r"^拍照[^。！？；]{0,120}(?:路由冲突|会话路由冲突|重试处理)[^。！？；]{0,120}[。！？；，、,\s]*$"
        ),
        "拍照暂时没成功，请稍后再试。",
    ),
    (
        re.compile(
            r"^现在请在设备端断开并重新进入一次实验会话[^。！？；]{0,120}[。！？；，、,\s]*$"
        ),
        "拍照暂时没成功，请稍后再试。",
    ),
]

BACKSTAGE_FULL_SENTENCE_PATTERNS = [
    re.compile(
        r"^这一步(?:先)?是[^。！？；]{0,120}"
        r"(?:总览确认|总览说明|总览阶段|概览确认)"
        r"[^。！？；]{0,160}"
        r"(?:记录什么|带你进入|共同准备阶段|共同准备的第一步)"
        r"[^。！？；]{0,120}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我先|我会先|我现在|我把|我这边先|我这边现在)"
        r"[^。！？；]{0,200}"
        r"(?:记上|写上|写入|记录|推进到|带你进入|进入共同准备阶段|进入共同准备的第一步|进入下一步|切到下一步)"
        r"[^。！？；]{0,200}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我先|我会先|我现在|我把|我这边先|我这边现在)"
        r"[^。！？；]{0,200}"
        r"(?:记上|写上|写入|记录)"
        r"[^。！？；]{0,120}"
        r"(?:带你做下一步|下一步共同加液|下一步共同操作|下一步共同准备)"
        r"[^。！？；]{0,160}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^这一步(?:先)?是[^。！？；]{0,120}"
        r"(?:我先查一下|我再查一下|我继续补一下|我补一下|我继续确认|我先确认)"
        r"[^。！？；]{0,220}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^这一步需要(?:把)?[^。！？；]{0,160}"
        r"(?:我继续补一下|我补一下|我先查一下|我继续确认|尽量把你现在要量的内容说准确|避免你按错量做)"
        r"[^。！？；]{0,220}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^现在(?:开始|进入)当前步骤[^。！？；]{0,120}"
        r"(?:我先|我会先|我现在)"
        r"[^。！？；]{0,200}"
        r"(?:确认|动作和记录要求|只带你做这一小步|只给你这一步|当前动作)"
        r"[^。！？；]{0,220}[。！？；，、,\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?我先查这一步在实验说明里的具体配法(?:，|,)?只告诉你现在要配的这一项[。！？；，\s]*$"
    ),
    re.compile(
        r"^这一步的浓度和用法我查到了(?:，|,)?我再核对一下实验说明里有没有写明具体配制量[。！？；，\s]*$"
    ),
    re.compile(
        r"^(?:只需要记|这一步只需要记)[^。！？；]{0,120}(?:不往这一步里乱写|不乱写)[。！？；，\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?我进入(?:[一二三四五]|[1-5])号样品(?:，|,)?只讲(?:[一二三四五]|[1-5])号现在该加什么[。！？；，\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我先|我现在|我继续)?(?:切到|进入)[^。！？；]{0,80}"
        r"(?:拍照这一步|拍照确认这一步|强制拍照这一步)"
        r"[^。！？；]{0,80}[。！？；，\s]*$"
    ),
    re.compile(
        r"^(?:我这边|这里)?(?:我现在|我这边现在)?执行(?:[一二三四五]|[1-5])号样品的(?:强制)?拍照"
        r"(?:，|,)?并把拍照结果写回当前步骤[。！？；，\s]*$"
    ),
    re.compile(
        r"^拍照已经成功(?:，|,)?我把拍照确认记录写回当前步骤并准备进入(?:[一二三四五]|[1-5])号样品[。！？；，\s]*$"
    ),
]


STRUCTURAL_BACKSTAGE_PATTERNS = [
    re.compile(
        "^(?:\\u6211\\u6309\\u4f60|\\u53ea\\u9700\\u8981\\u8bb0|\\u8fd9\\u4e00\\u6b65\\u53ea\\u9700\\u8981\\u8bb0)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,200}"
        "(?:\\u5199\\u5165|\\u8bb0\\u4e0a|\\u8bb0\\u5230|\\u4e0d\\u8865|\\u4e0d\\u5f80\\u8fd9\\u4e00\\u6b65\\u91cc\\u4e71\\u5199|\\u4e0d\\u4e71\\u5199)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u518d|\\u6211\\u5148|\\u6211\\u4f1a|\\u6211\\u73b0\\u5728|\\u6211\\u7ee7\\u7eed)"
        "(?:\\u786e\\u8ba4\\u4e00\\u4e0b|\\u786e\\u8ba4|\\u6838\\u5bf9\\u4e00\\u4e0b|\\u6838\\u5bf9)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8bb0\\u5f55\\u5b57\\u6bb5|\\u8bb0\\u5f55\\u8981\\u6c42|\\u5b57\\u6bb5)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,220}"
        "(?:\\u8865\\u62a5|\\u6f0f\\u62a5|\\u522b\\u7684\\u4fe1\\u606f|\\u522b\\u7684\\u5185\\u5bb9)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9|\\u8fd9\\u91cc)?(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u63a5\\u7740|\\u6211\\u7ee7\\u7eed)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,40}"
        "(?:\\u8bb0\\u4e0b|\\u8bb0\\u4e0b\\u6765|\\u8bb0\\u4e0a|\\u8bb0\\u5f55|\\u786e\\u8ba4)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u6700\\u7ec8\\u989c\\u8272|\\u989c\\u8272|\\u7a33\\u5b9a\\u65f6\\u95f4|\\u53cd\\u5e94\\u65f6\\u95f4|\\u8bb0\\u5f55\\u9879)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9|\\u8fd9\\u91cc)?(?:\\u6211\\u518d|\\u6211\\u63a5\\u7740|\\u6211\\u7ee7\\u7eed|\\u6211\\u5148)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,60}"
        "(?:\\u786e\\u8ba4|\\u6838\\u5bf9)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u8fd9\\u4e00\\u5c0f\\u6b65|\\u5f53\\u524d\\u8fd9\\u4e00\\u6b65|\\u8bb0\\u5f55\\u9879)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u53ea\\u8bb0|\\u53ea\\u786e\\u8ba4|\\u521a\\u624d\\u62a5\\u7684)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u8fd9\\u4e00\\u6b65\\u8bb0\\u5f55\\u9f50\\u4e86"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,200}"
        "(?:\\u7ed3\\u675f\\u5f53\\u524d\\u6b65\\u9aa4|\\u5207\\u5230\\u4e0b\\u4e00\\u6b65|\\u5171\\u540c\\u52a8\\u4f5c|\\u63a5\\u4e0b\\u6765\\u8981\\u505a\\u7684\\u5171\\u540c\\u52a8\\u4f5c)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u5148\\u628a(?:\\u8fd9\\u4e00\\u8f6e|\\u5f53\\u524d|\\u8fd9\\u4e00\\u6b65)?"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8bb0\\u5f55\\u5b8c\\u6210|\\u8bb0\\u5f55\\u505a\\u5b8c|\\u8bb0\\u5f55\\u9f50\\u4e86|\\u8bb0\\u5f55\\u7ed3\\u675f)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u518d\\u8fdb\\u5165|\\u518d\\u5207\\u5230|\\u7136\\u540e\\u8fdb\\u5165)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u4e0b\\u4e00\\u79cd\\u5171\\u540c\\u8bd5\\u5242|\\u4e0b\\u4e00\\u79cd\\u8bd5\\u5242|\\u5171\\u540c\\u8bd5\\u5242|\\u4e0b\\u4e00\\u8f6e\\u5171\\u540c\\u52a0\\u6db2|\\u5171\\u540c\\u52a8\\u4f5c)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9|\\u8fd9\\u91cc)?(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u73b0\\u5728|\\u6211\\u7ee7\\u7eed)"
        "\\u628a[^\\u3002\\uff01\\uff1f\\uff1b]{0,200}"
        "(?:\\u8bb0\\u4e0a|\\u5199\\u5165|\\u8bb0\\u5230|\\u5199\\u5230)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5b8c\\u6210\\u62cd\\u7167\\u786e\\u8ba4|\\u62cd\\u7167\\u786e\\u8ba4)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u4e0b\\u4e00\\u6b65\\u662f[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5f3a\\u5236\\u62cd\\u7167\\u786e\\u8ba4|\\u62cd\\u7167\\u786e\\u8ba4)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u6211\\u76f4\\u63a5\\u6267\\u884c|\\u6211\\u73b0\\u5728\\u6267\\u884c|\\u6211\\u9a6c\\u4e0a\\u6267\\u884c)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u62cd\\u597d\\u4e86(?:\\uff0c|,)?(?:\\u6211\\u628a)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u62cd\\u7167\\u786e\\u8ba4|\\u7167\\u7247\\u786e\\u8ba4)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8bb0\\u5230|\\u8bb0\\u4e0a|\\u5199\\u5230|\\u5199\\u5165)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9|\\u8fd9\\u91cc)?\\u6211\\u8fdb\\u5165(?:[\\u4e00\\u4e8c\\u4e09\\u56db\\u4e94]|[1-5])\\u53f7\\u6837\\u54c1(?:\\uff0c|,)?"
        "(?:\\u53ea\\u8bb2|\\u53ea\\u7ed9\\u4f60)[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8fd9\\u4e00\\u8f6e|\\u73b0\\u5728\\u8be5\\u52a0\\u4ec0\\u4e48|\\u4f53\\u79ef)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9|\\u8fd9\\u91cc)?\\u6211\\u5148\\u67e5[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5b9e\\u9a8c\\u8bf4\\u660e|\\u6587\\u6863)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u5177\\u4f53\\u914d\\u6cd5|\\u914d\\u5236\\u91cf|\\u8fd9\\u4e00\\u9879)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u8fd9\\u4e00\\u6b65\\u7684\\u6d53\\u5ea6\\u548c\\u7528\\u6cd5\\u6211\\u67e5\\u5230\\u4e86(?:\\uff0c|,)?"
        "\\u6211\\u518d\\u6838\\u5bf9\\u4e00\\u4e0b(?:\\u5b9e\\u9a8c\\u8bf4\\u660e|\\u6587\\u6863)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}(?:\\u5177\\u4f53\\u914d\\u5236\\u91cf|\\u5177\\u4f53\\u914d\\u6cd5)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u73b0\\u5728|\\u6211\\u5148|\\u6211\\u518d)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u542f\\u52a8|\\u5f00\\u59cb)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u6279\\u91cf\\u626b\\u63cf|\\u626b\\u63cf)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u53c2\\u6570\\u683c\\u5f0f\\u4e0d\\u5bf9(?:\\uff0c|,)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u91cd\\u8bd5|\\u6309\\u6b63\\u786e\\u683c\\u5f0f\\u91cd\\u8bd5)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6279\\u91cf\\u626b\\u63cf|\\u626b\\u63cf)[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "\\u8d85\\u65f6\\u4e86(?:\\uff0c|,)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,160}"
        "(?:\\u72b6\\u6001|\\u91cd\\u8bd5)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u626b\\u63cf\\u8bf7\\u6c42\\u5df2\\u7ecf\\u53d1\\u51fa|\\u8bf7\\u6c42\\u5df2\\u7ecf\\u53d1\\u51fa)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}(?:\\u7ed3\\u679c\\u8fd8\\u6ca1\\u8fd4\\u56de|\\u8fd8\\u6ca1\\u8fd4\\u56de)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u6211\\u7b49\\u4e00\\u4f1a\\u513f\\u518d\\u67e5\\u4e00\\u6b21[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u73b0\\u5728|\\u6211\\u518d|\\u6211\\u5148)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u91cd\\u65b0\\u67e5\\u8be2|\\u67e5\\u8be2)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}(?:\\u626b\\u63cf\\u6709\\u6ca1\\u6709\\u7ed3\\u675f|\\u6709\\u6ca1\\u6709\\u7ed3\\u675f)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^\\u8fde\\u63a5[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}(?:\\u65ad\\u4e86\\u4e00\\u4e0b|\\u65ad\\u5f00\\u4e86)"
        "(?:\\uff0c|,)?[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u91cd\\u8fde\\u5149\\u8c31\\u4eea|\\u786e\\u8ba4\\u72b6\\u6001)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u8fd9\\u8fb9)?\\u6682\\u65f6\\u53d6\\u4e0d\\u5230(?:\\u626b\\u63cf\\u7ed3\\u679c|\\u7ed3\\u679c)"
        "[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
]

TRANSITION_BACKSTAGE_PATTERNS = [
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u7ee7\\u7eed|\\u6211\\u73b0\\u5728|"
        "\\u6211\\u8fd9\\u8fb9\\u5148|\\u6211\\u8fd9\\u8fb9\\u73b0\\u5728)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u5e26\\u4f60\\u505a|\\u5e26\\u4f60\\u8fdb|\\u5e26\\u4f60\\u7ee7\\u7eed|"
        "\\u7ee7\\u7eed\\u5e26\\u4f60|\\u53ea\\u7ed9\\u4f60\\u8fd9\\u4e00\\u6b65|"
        "\\u53ea\\u5e26\\u4f60\\u505a)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5f53\\u524d|\\u8fd9\\u4e00\\u6b65|\\u4e0b\\u4e00\\u6b65)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u7ee7\\u7eed|\\u6211\\u73b0\\u5728|"
        "\\u6211\\u8fd9\\u8fb9\\u5148|\\u6211\\u8fd9\\u8fb9\\u73b0\\u5728)?"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u8bb0\\u5f55|\\u8865\\u9f50|\\u8bb0\\u4e0a|\\u5199\\u5165|\\u5199\\u4e0a|"
        "\\u63d0\\u4ea4|\\u8865\\u5f55|\\u5bf9\\u9f50)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8fd9\\u4e00\\u6b65|\\u5f53\\u524d|\\u5f53\\u524d\\u8fd9\\u4e00\\u6b65)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5e26\\u4f60|\\u518d\\u5e26\\u4f60|\\u7136\\u540e\\u5e26\\u4f60|"
        "\\u8fdb\\u4e0b\\u4e00\\u6b65|\\u505a\\u4e0b\\u4e00\\u6b65)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u7ee7\\u7eed|\\u6211\\u73b0\\u5728)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u786e\\u8ba4|\\u8bfb\\u4e00\\u4e0b|\\u770b\\u4e00\\u4e0b)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5f53\\u524d\\u8fd9\\u4e00\\u6b65|\\u8fd9\\u4e00\\u6b65|\\u5f53\\u524d)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8981\\u6c42|\\u8bb0\\u5f55|\\u7136\\u540e|\\u518d\\u7ee7\\u7eed|\\u518d\\u5e26\\u4f60)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u7ee7\\u7eed|\\u6211\\u73b0\\u5728)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,80}"
        "(?:\\u786e\\u8ba4|\\u8bfb\\u4e00\\u4e0b|\\u770b\\u4e00\\u4e0b|\\u68b3\\u7406\\u4e00\\u4e0b)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u5f53\\u524d\\u8fd9\\u4e00\\u6b65|\\u8fd9\\u4e00\\u6b65|\\u5f53\\u524d)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:"
        "\\u53ea\\u9700\\u8981\\u4f60\\u505a\\u4ec0\\u4e48(?:\\u3001|,|\\u548c)?\\u56de\\u62a5\\u4ec0\\u4e48|"
        "\\u53ea\\u9700\\u8981\\u4f60\\u56de\\u62a5\\u4ec0\\u4e48|"
        "\\u53ea\\u9700\\u8981\\u4f60\\u505a\\u4ec0\\u4e48|"
        "\\u505a\\u4ec0\\u4e48(?:\\u3001|,|\\u548c)?\\u56de\\u62a5\\u4ec0\\u4e48|"
        "\\u56de\\u62a5\\u4ec0\\u4e48"
        ")"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u73b0\\u5728|\\u6211\\u8fd9\\u8fb9\\u5148)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,60}"
        "(?:\\u8865\\u8bb0|\\u8bb0\\u4e00\\u4e0b|\\u8865\\u4e00\\u4e0b)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8fd9\\u4e00\\u8f6e\\u7ed3\\u679c|\\u8fd9\\u4e00\\u8f6e|\\u5f53\\u524d\\u7ed3\\u679c|\\u5f53\\u524d\\u8fd9\\u4e00\\u8f6e)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u73b0\\u5728|\\u6211\\u8fd9\\u8fb9\\u5148)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,60}"
        "(?:\\u6838\\u5bf9|\\u786e\\u8ba4|\\u770b\\u4e00\\u773c|\\u770b\\u4e00\\u4e0b)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u540e\\u9762\\u7684\\u7d27\\u63a5\\u6b65\\u9aa4|\\u540e\\u9762\\u7d27\\u63a5\\u6b65\\u9aa4|\\u540e\\u9762\\u6b65\\u9aa4|\\u7d27\\u63a5\\u6b65\\u9aa4|\\u4e0b\\u4e00\\u6b65)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
    re.compile(
        "^(?:\\u6211\\u5148|\\u6211\\u518d|\\u6211\\u73b0\\u5728|\\u6211\\u8fd9\\u8fb9\\u5148)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,60}"
        "(?:\\u770b\\u4e00\\u773c|\\u770b\\u4e00\\u4e0b|\\u786e\\u8ba4|\\u68b3\\u7406)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}"
        "(?:\\u8fd9\\u4e00\\u6b65\\u8981\\u4f60\\u56de\\u62a5\\u4ec0\\u4e48|\\u8fd9\\u4e00\\u6b65\\u8981\\u56de\\u62a5\\u4ec0\\u4e48|\\u8981\\u4f60\\u56de\\u62a5\\u4ec0\\u4e48|\\u8981\\u56de\\u62a5\\u4ec0\\u4e48)"
        "[^\\u3002\\uff01\\uff1f\\uff1b]{0,120}[\\u3002\\uff01\\uff1f\\uff1b\\uff0c\\s]*$"
    ),
]


def get_string_no_punctuation_or_emoji(s):
    """Strip leading/trailing whitespace, punctuation, and emoji."""
    chars = list(s or "")
    start = 0
    while start < len(chars) and is_punctuation_or_emoji(chars[start]):
        start += 1

    end = len(chars) - 1
    while end >= start and is_punctuation_or_emoji(chars[end]):
        end -= 1

    return "".join(chars[start : end + 1])


def is_punctuation_or_emoji(char):
    if char.isspace() or char in PUNCTUATION_SET:
        return True
    return is_emoji(char)


async def get_emotion(conn, text):
    """Send a simple emotion hint based on the first emoji found in text."""
    emoji = "😃"
    emotion = "happy"
    for char in text or "":
        if char in EMOJI_MAP:
            emoji = char
            emotion = EMOJI_MAP[char]
            break
    try:
        await conn.websocket.send(
            json.dumps(
                {
                    "type": "llm",
                    "text": emoji,
                    "emotion": emotion,
                    "session_id": conn.session_id,
                }
            )
        )
    except Exception as e:
        conn.logger.bind(tag=TAG).warning(f"发送情绪表情失败，错误:{e}")


def is_emoji(char):
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in EMOJI_RANGES)


def check_emoji(text):
    """Remove emoji and newlines from text before it is spoken."""
    return "".join(char for char in (text or "") if not is_emoji(char) and char != "\n")


def normalize_spoken_text(text):
    """Deterministic spoken-text normalization: collapse whitespace only."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


_DEFAULT_TTS_SPOKEN_ALIASES = {
    "AgNPs": "银纳米粒子",
    "AuNPs": "金纳米粒子",
    "H2O": "水",
    "H2O2": "过氧化氢",
    "HCl": "盐酸",
    "HBr": "氢溴酸",
    "HI": "氢碘酸",
    "HF": "氢氟酸",
    "HNO3": "硝酸",
    "H2SO4": "硫酸",
    "H2SO3": "亚硫酸",
    "H3PO4": "磷酸",
    "NaBH4": "硼氢化钠",
    "Na3Cit": "柠檬酸钠",
    "NH3·H2O": "氨水",
    "NH4OH": "氨水",
    "CH3COOH": "乙酸",
    "C2H5OH": "乙醇",
    "EtOH": "乙醇",
    "CH3OH": "甲醇",
    "MeOH": "甲醇",
    "IPA": "异丙醇",
    "4-NP": "4-硝基苯酚",
    "4-AP": "4-氨基苯酚",
    "PVP": "聚乙烯吡咯烷酮",
    "PEG": "聚乙二醇",
    "CTAB": "十六烷基三甲基溴化铵",
    "SDS": "十二烷基硫酸钠",
}

_CHEMICAL_ELEMENT_NAMES = {
    "H": "氢",
    "He": "氦",
    "Li": "锂",
    "Be": "铍",
    "B": "硼",
    "C": "碳",
    "N": "氮",
    "O": "氧",
    "F": "氟",
    "Ne": "氖",
    "Na": "钠",
    "Mg": "镁",
    "Al": "铝",
    "Si": "硅",
    "P": "磷",
    "S": "硫",
    "Cl": "氯",
    "Ar": "氩",
    "K": "钾",
    "Ca": "钙",
    "Sc": "钪",
    "Ti": "钛",
    "V": "钒",
    "Cr": "铬",
    "Mn": "锰",
    "Fe": "铁",
    "Co": "钴",
    "Ni": "镍",
    "Cu": "铜",
    "Zn": "锌",
    "Ga": "镓",
    "Ge": "锗",
    "As": "砷",
    "Se": "硒",
    "Br": "溴",
    "Kr": "氪",
    "Rb": "铷",
    "Sr": "锶",
    "Ag": "银",
    "Cd": "镉",
    "Sn": "锡",
    "Sb": "锑",
    "I": "碘",
    "Ba": "钡",
    "Pt": "铂",
    "Au": "金",
    "Hg": "汞",
    "Pb": "铅",
    "Bi": "铋",
}

_CHEMICAL_GROUP_SALT_STEMS = {
    "OH": "氢氧化",
    "O2": "过氧化",
    "NO2": "亚硝酸",
    "NO3": "硝酸",
    "SO3": "亚硫酸",
    "HSO3": "亚硫酸氢",
    "SO4": "硫酸",
    "HSO4": "硫酸氢",
    "CO3": "碳酸",
    "HCO3": "碳酸氢",
    "PO4": "磷酸",
    "HPO4": "磷酸氢",
    "H2PO4": "磷酸二氢",
    "MnO4": "高锰酸",
    "CrO4": "铬酸",
    "Cr2O7": "重铬酸",
    "S2O3": "硫代硫酸",
    "CH3COO": "乙酸",
    "C2H3O2": "乙酸",
    "HCOO": "甲酸",
    "C2O4": "草酸",
    "C6H5O7": "柠檬酸",
    "ClO": "次氯酸",
    "ClO2": "亚氯酸",
    "ClO3": "氯酸",
    "ClO4": "高氯酸",
    "BrO3": "溴酸",
    "IO3": "碘酸",
    "CN": "氰化",
    "SCN": "硫氰酸",
}

_BINARY_ANION_STEMS = {
    "F": "氟化",
    "Cl": "氯化",
    "Br": "溴化",
    "I": "碘化",
    "O": "氧化",
    "S": "硫化",
    "N": "氮化",
    "P": "磷化",
    "H": "氢化",
    "C": "碳化",
}

_CHEMICAL_FORMULA_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9().·+\-]{1,})(?![A-Za-z0-9])"
)


_ELEMENT_PARENTHESES_ALIAS_RE = re.compile(
    r"\b([A-Z][a-z]?)\s*[（(]\s*([\u4e00-\u9fff]{1,4})\s*[)）]"
)
_ELEMENT_PREFIXED_NANOSTRUCTURE_RE = re.compile(
    r"\b([A-Z][a-z]?)\s*(纳米(?:粒子|颗粒|棒|线|片|花|球|立方体)|量子点)\b"
)
_AROMATIC_POSITION_PREFIX_MAP = {"2": "邻", "3": "间", "4": "对"}
_AROMATIC_POSITION_RE = re.compile(
    r"(?<![\dA-Za-z])([234])\s*[-－—–]\s*"
    r"([\u4e00-\u9fff]{1,24}"
    r"(?:苯酚|苯胺|苯甲酸|甲苯|甲酚|苯乙烯|苯腈|吡啶|联苯|萘|苯|酚|胺|酸|醛|酮|酯|腈|醚)"
    r"(?:[\u4e00-\u9fff]{0,4})?)"
    r"(?=(?:的|在|与|及|和|,|，|。|；|;|\s|$))"
)


def _count_to_chinese(count: int) -> str:
    mapping = {
        0: "零",
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    if count in mapping:
        return mapping[count]
    if count < 20:
        return "十" + mapping[count % 10]
    if count < 100:
        tens, ones = divmod(count, 10)
        text = mapping[tens] + "十"
        if ones:
            text += mapping[ones]
        return text
    return str(count)


def _with_stoich_prefix(stem: str, count: int) -> str:
    if count <= 1:
        return stem
    return f"{_count_to_chinese(count)}{stem}"


def _parse_simple_cation(formula: str):
    token = str(formula or "").strip()
    if not token:
        return None

    ammonium_match = re.fullmatch(r"(NH4)(\d*)", token)
    if ammonium_match:
        count_text = ammonium_match.group(2)
        return "铵", int(count_text or "1")

    element_match = re.fullmatch(r"([A-Z][a-z]?)(\d*)", token)
    if not element_match:
        return None

    symbol = element_match.group(1)
    if symbol not in _CHEMICAL_ELEMENT_NAMES:
        return None
    count_text = element_match.group(2)
    return _CHEMICAL_ELEMENT_NAMES[symbol], int(count_text or "1")


def _replace_range_for_tts(text: str) -> str:
    patterns = (
        r"(\d+)\s*[-~—–至到]+\s*(\d+)\s*号样品",
        r"(\d+)\s*[-~—–至到]+\s*(\d+)\s*号样本",
        r"(\d+)\s*[-~—–至到]+\s*(\d+)\s*号烧杯",
        r"(\d+)\s*[-~—–至到]+\s*(\d+)\s*号试管",
        r"(\d+)\s*[-~—–至到]+\s*(\d+)\s*组",
    )

    def _repl(match: re.Match) -> str:
        start = match.group(1)
        end = match.group(2)
        suffix = match.group(0)[match.end(2) - match.start(0) :]
        suffix = re.sub(r"^\s*", "", suffix)
        return f"{start}到{end}{suffix}"

    result = text
    for pattern in patterns:
        result = re.sub(pattern, _repl, result)
    return result


def _replace_measurement_ranges_for_tts(text: str) -> str:
    suffixes = (
        "转每分钟",
        "千赫兹",
        "摄氏度",
        "分钟",
        "小时",
        "毫升",
        "微升",
        "毫克",
        "微克",
        "微米",
        "毫米",
        "厘米",
        "赫兹",
        "毫伏",
        "毫安",
        "纳米",
        "秒",
        "天",
        "周",
        "月",
        "年",
        "升",
        "克",
        "伏",
        "安",
        "瓦",
        "度",
        "滴",
        "次",
        "倍",
        "圈",
        "轮",
        "档",
        "级",
        "份",
        "组",
        "步",
        "%",
        "％",
    )
    suffix_pattern = "|".join(re.escape(suffix) for suffix in suffixes)
    range_re = re.compile(
        rf"(第?)(\d+(?:\.\d+)?)\s*(?:-|~|～|—|–|至|到)\s*(\d+(?:\.\d+)?)\s*({suffix_pattern})"
    )

    def _repl(match: re.Match) -> str:
        prefix = match.group(1) or ""
        start = match.group(2)
        end = match.group(3)
        suffix = match.group(4)
        return f"{prefix}{start}到{end}{suffix}"

    return range_re.sub(_repl, text)


def _replace_units_for_tts(text: str) -> str:
    replacements = (
        (r"(?i)\bmmol\s*/\s*L\b", "毫摩尔每升"),
        (r"(?i)\bμmol\s*/\s*L\b", "微摩尔每升"),
        (r"(?i)\bµmol\s*/\s*L\b", "微摩尔每升"),
        (r"(?i)\bumol\s*/\s*L\b", "微摩尔每升"),
        (r"(?i)\bmol\s*/\s*L\b", "摩尔每升"),
        (r"(?i)\bmmol\s*·\s*L-?1\b", "毫摩尔每升"),
        (r"(?i)\bμmol\s*·\s*L-?1\b", "微摩尔每升"),
        (r"(?i)\bµmol\s*·\s*L-?1\b", "微摩尔每升"),
        (r"(?i)\bumol\s*·\s*L-?1\b", "微摩尔每升"),
        (r"(?i)\bmol\s*·\s*L-?1\b", "摩尔每升"),
        (r"(?i)\bmg\s*/\s*mL\b", "毫克每毫升"),
        (r"(?i)\bg\s*/\s*L\b", "克每升"),
        (r"(?i)\bwt%\b", "质量百分比"),
        (r"(?i)\bvol%\b", "体积百分比"),
        (r"(?i)(?<=\d)\s*μL\b", "微升"),
        (r"(?i)(?<=\d)\s*µL\b", "微升"),
        (r"(?i)(?<=\d)\s*uL\b", "微升"),
        (r"(?i)(?<=\d)\s*mL\b", "毫升"),
        (r"(?i)(?<=\d)\s*L\b", "升"),
        (r"(?i)(?<=\d)\s*mg\b", "毫克"),
        (r"(?i)(?<=\d)\s*μg\b", "微克"),
        (r"(?i)(?<=\d)\s*µg\b", "微克"),
        (r"(?i)(?<=\d)\s*ug\b", "微克"),
        (r"(?i)(?<=\d)\s*g\b", "克"),
        (r"(?i)(?<=\d)\s*nm\b", "纳米"),
        (r"(?i)(?<![0-9A-Za-z_])nm(?![0-9A-Za-z_])", "纳米"),
        (r"(?i)(?<=\d)\s*μm\b", "微米"),
        (r"(?i)(?<=\d)\s*µm\b", "微米"),
        (r"(?i)(?<=\d)\s*mm\b", "毫米"),
        (r"(?i)(?<=\d)\s*cm\b", "厘米"),
        (r"(?i)(?<=\d)\s*kHz\b", "千赫兹"),
        (r"(?i)(?<=\d)\s*Hz\b", "赫兹"),
        (r"(?i)(?<=\d)\s*min\b", "分钟"),
        (r"(?i)(?<=\d)\s*mins\b", "分钟"),
        (r"(?i)(?<=\d)\s*sec\b", "秒"),
        (r"(?i)(?<=\d)\s*s\b", "秒"),
        (r"(?i)(?<=\d)\s*h\b", "小时"),
        (r"(?i)(?<=\d)\s*rpm\b", "转每分钟"),
        (r"(?i)(?<=\d)\s*V\b", "伏"),
        (r"(?i)(?<=\d)\s*mV\b", "毫伏"),
        (r"(?i)(?<=\d)\s*A\b", "安"),
        (r"(?i)(?<=\d)\s*mA\b", "毫安"),
        (r"(?i)(?<=\d)\s*W\b", "瓦"),
        (r"(?i)(?<=\d)\s*°C\b", "摄氏度"),
        (r"(?<=\d)\s*℃", "摄氏度"),
    )

    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)
    return result


def _replace_parenthesized_element_aliases_for_tts(text: str) -> str:
    def _repl(match: re.Match) -> str:
        symbol = match.group(1)
        alias = match.group(2).strip()
        element_name = _CHEMICAL_ELEMENT_NAMES.get(symbol)
        if not element_name:
            return match.group(0)
        if alias == element_name or alias in element_name or element_name in alias:
            return element_name
        return match.group(0)

    return _ELEMENT_PARENTHESES_ALIAS_RE.sub(_repl, text)


def _replace_element_prefixed_nanostructure_terms_for_tts(text: str) -> str:
    def _repl(match: re.Match) -> str:
        symbol = match.group(1)
        structure = match.group(2)
        element_name = _CHEMICAL_ELEMENT_NAMES.get(symbol)
        if not element_name:
            return match.group(0)
        return f"{element_name}{structure}"

    return _ELEMENT_PREFIXED_NANOSTRUCTURE_RE.sub(_repl, text)


def _replace_generic_nanostructure_aliases_for_tts(text: str) -> str:
    patterns = (
        (re.compile(r"\b([A-Z][a-z]?)NPs?\b"), "纳米粒子"),
        (re.compile(r"\b([A-Z][a-z]?)NRs?\b"), "纳米棒"),
        (re.compile(r"\b([A-Z][a-z]?)NWs?\b"), "纳米线"),
        (re.compile(r"\b([A-Z][a-z]?)NSs?\b"), "纳米片"),
        (re.compile(r"\b([A-Z][a-z]?)QDs?\b"), "量子点"),
    )

    result = text
    for pattern, suffix in patterns:
        def _repl(match: re.Match) -> str:
            symbol = match.group(1)
            element_name = _CHEMICAL_ELEMENT_NAMES.get(symbol)
            if not element_name:
                return match.group(0)
            return f"{element_name}{suffix}"

        result = pattern.sub(_repl, result)
    return result


def _replace_aromatic_position_for_tts(text: str) -> str:
    def _repl(match: re.Match) -> str:
        prefix = _AROMATIC_POSITION_PREFIX_MAP.get(match.group(1))
        if not prefix:
            return match.group(0)
        return f"{prefix}{match.group(2)}"

    return _AROMATIC_POSITION_RE.sub(_repl, text)


def _replace_formula_aliases_for_tts(text: str, custom_aliases=None) -> str:
    merged_aliases = dict(_DEFAULT_TTS_SPOKEN_ALIASES)
    if isinstance(custom_aliases, dict):
        for key, value in custom_aliases.items():
            key_text = str(key or "").strip()
            value_text = str(value or "").strip()
            if key_text and value_text:
                merged_aliases[key_text] = value_text

    result = text
    for source, target in sorted(
        merged_aliases.items(), key=lambda item: len(item[0]), reverse=True
    ):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])"
        result = re.sub(pattern, target, result)
    return result


def _is_formula_part_structure(part: str) -> bool:
    formula_part = str(part or "").strip()
    if not formula_part:
        return False

    def _parse(index: int, stop_char=None):
        saw_token = False
        while index < len(formula_part):
            current = formula_part[index]
            if stop_char and current == stop_char:
                return saw_token, index + 1
            if current == "(":
                inner_ok, next_index = _parse(index + 1, ")")
                if not inner_ok:
                    return False, next_index
                index = next_index
                while index < len(formula_part) and formula_part[index].isdigit():
                    index += 1
                saw_token = True
                continue
            if not current.isupper():
                return False, index

            symbol = current
            index += 1
            if index < len(formula_part) and formula_part[index].islower():
                symbol += formula_part[index]
                index += 1
            if symbol not in _CHEMICAL_ELEMENT_NAMES:
                return False, index
            while index < len(formula_part) and formula_part[index].isdigit():
                index += 1
            saw_token = True

        if stop_char:
            return False, index
        return saw_token, index

    parsed_ok, final_index = _parse(0)
    return parsed_ok and final_index == len(formula_part)


def _looks_like_chemical_formula(token: str) -> bool:
    core = str(token or "").strip("+-")
    if not core or not any(ch.isupper() for ch in core):
        return False
    if not (
        any(ch.isdigit() for ch in core)
        or "(" in core
        or ")" in core
        or "·" in core
        or "." in core
        or re.search(r"[a-z]", core)
    ):
        return False

    parts = re.split(r"[·.]", core)
    for part in parts:
        stripped_part = re.sub(r"^\d+", "", part)
        if not _is_formula_part_structure(stripped_part):
            return False
    return True


def _maybe_convert_hydrate_formula(formula: str):
    hydrate_match = re.fullmatch(r"(.+?)[·.]((\d+)?)H2O", formula)
    if not hydrate_match:
        return None

    base_formula = hydrate_match.group(1)
    hydrate_count = int(hydrate_match.group(3) or "1")
    base_spoken = _maybe_convert_formula_for_tts(base_formula)
    if not base_spoken:
        return None

    if hydrate_count <= 1:
        return f"{base_spoken}水合物"
    return f"{base_spoken}{_count_to_chinese(hydrate_count)}水合物"


def _maybe_convert_parenthesized_salt_formula(formula: str):
    match = re.fullmatch(r"((?:NH4)|(?:[A-Z][a-z]?))(\d*)\(([^()]+)\)(\d*)", formula)
    if not match:
        return None

    cation_formula = f"{match.group(1)}{match.group(2)}"
    anion_formula = match.group(3)
    cation = _parse_simple_cation(cation_formula)
    anion_stem = _CHEMICAL_GROUP_SALT_STEMS.get(anion_formula)
    if not cation or not anion_stem:
        return None

    cation_name, _ = cation
    return f"{anion_stem}{cation_name}"


def _maybe_convert_simple_salt_formula(formula: str):
    cation = None
    anion_stem = None

    for group_formula in sorted(
        _CHEMICAL_GROUP_SALT_STEMS.keys(), key=len, reverse=True
    ):
        if not formula.endswith(group_formula):
            continue
        cation = _parse_simple_cation(formula[: -len(group_formula)])
        if not cation:
            continue
        anion_stem = _CHEMICAL_GROUP_SALT_STEMS[group_formula]
        break

    if not cation or not anion_stem:
        for anion_symbol, binary_stem in sorted(
            _BINARY_ANION_STEMS.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if not formula.endswith(anion_symbol):
                continue
            cation = _parse_simple_cation(formula[: -len(anion_symbol)])
            if not cation:
                continue
            anion_stem = binary_stem
            break

    if not cation or not anion_stem:
        return None

    cation_name, _ = cation
    return f"{anion_stem}{cation_name}"


def _maybe_convert_binary_formula(formula: str):
    match = re.fullmatch(r"([A-Z][a-z]?)(\d*)([A-Z][a-z]?)(\d*)", formula)
    if not match:
        return None

    cation_symbol = match.group(1)
    cation_count = int(match.group(2) or "1")
    anion_symbol = match.group(3)
    anion_count = int(match.group(4) or "1")

    cation_name = _CHEMICAL_ELEMENT_NAMES.get(cation_symbol)
    anion_stem = _BINARY_ANION_STEMS.get(anion_symbol)
    if not cation_name or not anion_stem:
        return None

    spoken_anion = _with_stoich_prefix(anion_stem, anion_count)
    spoken_cation = _with_stoich_prefix(cation_name, cation_count)
    return f"{spoken_anion}{spoken_cation}"


def _maybe_convert_formula_for_tts(formula: str):
    token = str(formula or "").strip()
    if not token or not _looks_like_chemical_formula(token):
        return None

    return (
        _maybe_convert_hydrate_formula(token)
        or _maybe_convert_parenthesized_salt_formula(token)
        or _maybe_convert_simple_salt_formula(token)
        or _maybe_convert_binary_formula(token)
    )


def _replace_generic_formulae_for_tts(text: str) -> str:
    def _repl(match: re.Match) -> str:
        token = match.group(1)
        replacement = _maybe_convert_formula_for_tts(token)
        if replacement:
            return replacement
        return token

    return _CHEMICAL_FORMULA_TOKEN_RE.sub(_repl, text)


def normalize_tts_text(text, custom_aliases=None):
    """Normalize final spoken text for Chinese TTS pronunciation."""
    normalized = normalize_spoken_text(text)
    if not normalized:
        return ""

    normalized = _replace_range_for_tts(normalized)
    normalized = _replace_numbered_hao_labels_for_tts(normalized)
    normalized = _replace_units_for_tts(normalized)
    normalized = _replace_measurement_ranges_for_tts(normalized)
    normalized = _replace_parenthesized_element_aliases_for_tts(normalized)
    normalized = _replace_element_prefixed_nanostructure_terms_for_tts(normalized)
    normalized = _replace_generic_nanostructure_aliases_for_tts(normalized)
    normalized = _replace_formula_aliases_for_tts(
        normalized, custom_aliases=custom_aliases
    )
    normalized = _replace_generic_formulae_for_tts(normalized)
    normalized = _replace_aromatic_position_for_tts(normalized)
    normalized = _replace_decimal_numbers_for_tts(normalized)
    normalized = normalized.replace("×", "乘")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


CANONICAL_OPENING_RE = re.compile(
    r"今天我们做《[^》\n]{1,80}》。你准备好开始了吗[？?]"
)
_DIGIT_TO_CHINESE = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}
_SPOKEN_DECIMAL_RE = re.compile(
    r"(?<![0-9A-Za-z_-])([+-]?\d+)\.(\d+)(?![0-9A-Za-z_.-])"
)
_SPOKEN_HAO_RANGE_RE = re.compile(
    r"(?<![0-9A-Za-z_.-])([+-]?\d+)\s*(?:到|至)\s*([+-]?\d+)\s*号"
)
_SPOKEN_HAO_INDEX_RE = re.compile(r"(?<![0-9A-Za-z_.-])([+-]?\d+)\s*号")
_SPOKEN_URL_RE = re.compile(r"\b(?:https?|wss?)://\S+", re.IGNORECASE)
_SPOKEN_WINDOWS_PATH_RE = re.compile(
    r"(?<!\w)(?:[A-Za-z]:\\|\\\\)[^\s，。！？；]+"
)
_SPOKEN_FILE_NAME_RE = re.compile(
    r"\b[^\s\\/]+\.(?:pdf|ya?ml|json|csv|png|jpe?g|wav)\b",
    re.IGNORECASE,
)
_SPOKEN_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_SPOKEN_TECHNICAL_FIELD_RE = re.compile(
    r"\b(?:device_id|session_id|chat_session_id|transport_session_id|"
    r"connection_session_id|model_session_key|local_path|file_path|output_path|"
    r"yaml_path|pdf_path|photo_path|task_id|client_id|tool_name|function_name|"
    r"session_key|ready_for_samples|sample_positions|wavelength_nm|duration_minutes|"
    r"interval_seconds|run_name|sessionkey|readyforsamples|samplepositions|"
    r"jsonrpc|serverinfo|capabilities)\b"
    r"\s*[:=：]?\s*[^\s，。！？；]*",
    re.IGNORECASE,
)
_SPOKEN_TOOL_NAME_RE = re.compile(
    r"\b(?:create_session|get_overview|list_steps|get_step|get_schema|"
    r"get_current_progress|get_progress_summary|start_trial|add_field|add_fields|"
    r"finish_trial|can_proceed|proceed_to_next_step|get_modifiable_records|"
    r"modify_record|redo_trial|redirect_to_step|cancel_trial|"
    r"get_experiment_reference|search_experiment_reference|export_records|"
    r"export_records_to_yaml|xiaozhi_[a-z_]+|uvvis_[a-z_]+|self_[a-z_]+)\b"
)
_SPOKEN_TOOL_ARGUMENT_BLOCK_RE = re.compile(
    r"\(\s*(?:sample_positions|samplepositions|ready_for_samples|readyforsamples|"
    r"session_key|sessionkey|wavelength_nm|duration_minutes|interval_seconds|run_name)"
    r"[^)]*\)",
    re.IGNORECASE,
)
_SPOKEN_TRAILING_TECHNICAL_TAIL_RE = re.compile(
    r"(?:[，,、 ]*(?:路径|位置|地址)\s*(?:是|为|在)?|"
    r"[，,、 ]*(?:已保存为|保存为|保存到|保存在|输出到|位于))\s*$"
)


def _collapse_duplicated_canonical_opening(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""

    opening_match = CANONICAL_OPENING_RE.match(normalized)
    if not opening_match:
        return normalized

    opening = opening_match.group(0)
    opening_without_q = re.sub(r"[？?]\s*$", "", opening)
    repeated_opening_re = re.compile(
        rf"^(?:{re.escape(opening)}\s*)+(?:{re.escape(opening_without_q)}\s*)?$"
    )
    if repeated_opening_re.fullmatch(normalized):
        return opening

    return normalized


def _section_to_chinese(section: int) -> str:
    digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千"]
    result = []
    zero_pending = False
    remaining = int(section)

    for power in range(3, -1, -1):
        divisor = 10**power
        digit = remaining // divisor
        remaining %= divisor
        if digit == 0:
            if result:
                zero_pending = True
            continue
        if zero_pending:
            result.append("零")
            zero_pending = False
        if not (digit == 1 and power == 1 and not result):
            result.append(digits[digit])
        result.append(units[power])
    return "".join(result) or "零"


def _integer_to_chinese(num_text: str) -> str:
    try:
        number = int(str(num_text or "").strip())
    except Exception:
        return str(num_text or "")

    if number == 0:
        return "零"

    negative = number < 0
    if negative:
        number = abs(number)

    big_units = ["", "万", "亿", "兆"]
    sections = []
    while number > 0:
        sections.append(number % 10000)
        number //= 10000

    parts = []
    need_zero = False
    for index in range(len(sections) - 1, -1, -1):
        section = sections[index]
        if section == 0:
            need_zero = bool(parts)
            continue
        if parts and (need_zero or section < 1000):
            parts.append("零")
        parts.append(_section_to_chinese(section) + big_units[index])
        need_zero = False

    result = re.sub(r"零+", "零", "".join(parts)).rstrip("零")
    return f"负{result}" if negative else result


def _replace_decimal_numbers_for_tts(text: str) -> str:
    def _repl(match: re.Match) -> str:
        integer_part = _integer_to_chinese(match.group(1))
        fractional_part = "".join(
            _DIGIT_TO_CHINESE.get(ch, ch) for ch in match.group(2)
        )
        return f"{integer_part}点{fractional_part}"

    return _SPOKEN_DECIMAL_RE.sub(_repl, text)


def _replace_numbered_hao_labels_for_tts(text: str) -> str:
    def _range_repl(match: re.Match) -> str:
        start = _integer_to_chinese(match.group(1))
        end = _integer_to_chinese(match.group(2))
        return f"{start}到{end}号"

    normalized = _SPOKEN_HAO_RANGE_RE.sub(_range_repl, text)

    def _index_repl(match: re.Match) -> str:
        number = _integer_to_chinese(match.group(1))
        return f"{number}号"

    return _SPOKEN_HAO_INDEX_RE.sub(_index_repl, normalized)


def _split_spoken_sentence_chunks(text: str):
    chunks = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", str(text or ""))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _is_spoken_meta_guidance_clause(text: str) -> bool:
    clause = normalize_spoken_text(text).strip(" ，,、；;。！？!?")
    if not clause:
        return False

    scope_prefixes = (
        "只完成",
        "只给",
        "只讲",
        "只提醒",
        "只说",
        "只告诉",
        "只推进",
        "只按",
        "只需要完成",
        "只需要做",
    )
    scope_topics = (
        "确认",
        "共同试剂",
        "后续",
        "下一步",
        "这一步",
        "当前步骤",
        "本步",
        "主说话人",
        "当前动作",
        "同义表达",
    )
    if clause.startswith(scope_prefixes) and any(token in clause for token in scope_topics):
        return True

    negative_prefixes = ("不要", "也不要", "别", "不用", "不需要")
    meta_verbs = ("重复", "讲", "说", "提", "展开", "预告", "复述", "补充", "介绍")
    meta_topics = (
        "共同试剂",
        "后续",
        "后续加液",
        "下一步",
        "后面的步骤",
        "背景",
        "原理",
        "记录字段",
        "字段",
        "流程总览",
    )
    if clause.startswith(negative_prefixes):
        if any(token in clause for token in meta_verbs) and any(
            token in clause for token in meta_topics
        ):
            return True

    explicit_meta_clauses = (
        "不要重复共同试剂",
        "不需要重复共同试剂",
        "也不要讲后续",
        "不要讲后续",
        "不要说后续",
        "不需要讲后续",
        "不要预告下一步",
        "只给主说话人当前动作",
    )
    return any(token in clause for token in explicit_meta_clauses)


def _strip_spoken_internal_control_tail(text: str) -> str:
    clause = normalize_spoken_text(text).strip(" ，,、；;。！？!?")
    if not clause:
        return ""

    rewritten = clause
    lead_in_patterns = (
        re.compile(r"^(?:先|再)?提示主说话人[“\"']?"),
        re.compile(r"^(?:先|再)?只需提醒主说话人"),
        re.compile(r"^(?:先|再)?提醒主说话人"),
    )
    for pattern in lead_in_patterns:
        rewritten = pattern.sub("", rewritten).strip(" “”\"'")

    tail_patterns = (
        re.compile(
            r"(?:，|,)?(?:只有在|只有当)[^。！？!?；;]{0,160}"
            r"(?:主说话人|同义表达|pure water|liquid blank|记录纯水空白|记录液体空白|当前批次|返回结果|session_key|uvvis_measure)"
            r"[^。！？!?；;]{0,240}$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:，|,)?(?:先|再)?根据上一步[^。！？!?；;]{0,240}"
            r"(?:返回结果|pure water|liquid blank|记录纯水空白|记录液体空白|可复用|调用|不要再调用)"
            r"[^。！？!?；;]{0,240}$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:，|,)?(?:先|再)?根据[^。！？!?；;]{0,60}返回结果[^。！？!?；;]{0,240}"
            r"(?:pure water|liquid blank|可复用|调用|记录)"
            r"[^。！？!?；;]{0,240}$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:，|,)?(?:若|如果)工具(?:提示|返回)[^。！？!?；;]{0,240}"
            r"(?:可复用|不要重复测量|读取并记住|pure water|liquid blank|返回结果)"
            r"[^。！？!?；;]{0,240}$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:，|,)?(?:但|并且|并|然后)?[^。！？!?；;]{0,60}"
            r"(?:读取并记住|不要再调用|才调用)"
            r"[^。！？!?；;]{0,240}$",
            re.IGNORECASE,
        ),
    )
    for pattern in tail_patterns:
        rewritten = pattern.sub("", rewritten).strip(" ，,、；;")

    return rewritten.strip(" “”\"'")


def _strip_spoken_meta_guidance_clauses(text: str) -> str:
    cleaned_sentences = []
    for sentence in _split_spoken_sentence_chunks(text):
        stripped = sentence.strip()
        terminal = stripped[-1] if stripped and stripped[-1] in "。！？!?；;" else ""
        body = stripped[:-1] if terminal else stripped
        clauses = [part.strip() for part in re.split(r"[，,；;]\s*", body) if part.strip()]
        kept_clauses = []
        for clause in clauses:
            rewritten_clause = _strip_spoken_internal_control_tail(clause)
            if not rewritten_clause:
                continue
            if _is_spoken_meta_guidance_clause(rewritten_clause):
                continue
            kept_clauses.append(rewritten_clause)
        if not kept_clauses:
            continue
        rebuilt = "，".join(kept_clauses).strip(" ，,、；;")
        if not rebuilt:
            continue
        if terminal:
            rebuilt += terminal
        cleaned_sentences.append(rebuilt)
    return "".join(cleaned_sentences).strip()


def _is_spoken_future_step_clause(text: str) -> bool:
    clause = normalize_spoken_text(text).strip(" ，,、；;。！？!?")
    if not clause:
        return False

    normalized = re.sub(r"\s+", "", clause)
    transition_patterns = (
        re.compile(
            r"(?:完成|做完|结束)(?:这一轮|本轮|当前步骤|当前这一步|这一步|这步|本步)?"
            r"(?:后|之后|以后).{0,24}(?:再|然后)?.{0,12}(?:回到|进入|开始|继续)"
        ),
        re.compile(
            r"(?:当前步骤|当前这一步|这一步|这步|本步|这一轮|本轮)结束后"
            r".{0,24}(?:回到|进入|开始|继续)"
        ),
        re.compile(
            r"(?:回到|进入)[^。！？；]{0,24}(?:号样品|样品|步骤|阶段)"
            r"[^。！？；]{0,24}(?:后续|下一步|继续|搅拌|加液|操作)"
        ),
    )
    return any(pattern.search(normalized) for pattern in transition_patterns)


def _strip_spoken_future_step_clauses(text: str) -> str:
    cleaned_sentences = []
    for sentence in _split_spoken_sentence_chunks(text):
        stripped = sentence.strip()
        terminal = stripped[-1] if stripped and stripped[-1] in "。！？!?；;" else ""
        body = stripped[:-1] if terminal else stripped
        clauses = [part.strip() for part in re.split(r"[，,；;]\s*", body) if part.strip()]
        kept_clauses = [
            clause for clause in clauses if not _is_spoken_future_step_clause(clause)
        ]
        if not kept_clauses:
            continue
        rebuilt = "，".join(kept_clauses).strip(" ，,、；;")
        if not rebuilt:
            continue
        if terminal:
            rebuilt += terminal
        cleaned_sentences.append(rebuilt)
    return "".join(cleaned_sentences).strip()


def _strip_spoken_technical_details(text: str) -> str:
    cleaned_sentences = []
    for sentence in _split_spoken_sentence_chunks(text):
        cleaned = sentence
        cleaned = _SPOKEN_INLINE_CODE_RE.sub("", cleaned)
        cleaned = _SPOKEN_URL_RE.sub("", cleaned)
        cleaned = _SPOKEN_WINDOWS_PATH_RE.sub("", cleaned)
        cleaned = _SPOKEN_TECHNICAL_FIELD_RE.sub("", cleaned)
        cleaned = _SPOKEN_TOOL_NAME_RE.sub("", cleaned)
        cleaned = _SPOKEN_TOOL_ARGUMENT_BLOCK_RE.sub("", cleaned)
        cleaned = _SPOKEN_FILE_NAME_RE.sub("", cleaned)
        cleaned = _SPOKEN_TRAILING_TECHNICAL_TAIL_RE.sub("", cleaned)
        cleaned = re.sub(r"[，,、]+\s*([。！？!?；;])", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,、:：；;")
        cleaned = re.sub(r"[，,、]{2,}", "，", cleaned)
        if not cleaned:
            continue
        if cleaned[-1] not in "。！？!?；;":
            cleaned += "。"
        visible = get_string_no_punctuation_or_emoji(cleaned)
        if not visible:
            continue
        cleaned_sentences.append(cleaned)
    return "".join(cleaned_sentences).strip()


def _limit_spoken_sentence_count(text: str, max_sentences: int = 2) -> str:
    sentences = _split_spoken_sentence_chunks(text)
    if len(sentences) <= max_sentences:
        return "".join(sentences).strip()
    if max_sentences <= 1:
        return sentences[0].strip()

    kept = list(sentences[: max_sentences - 1])
    tail_parts = []
    tail_sentences = sentences[max_sentences - 1 :]
    for index, sentence in enumerate(tail_sentences):
        segment = sentence.strip()
        if index < len(tail_sentences) - 1:
            segment = segment.rstrip("。！？!?；;，,、 ")
        tail_parts.append(segment)

    merged_tail = "，".join(part for part in tail_parts if part).strip("，,、 ")
    if merged_tail and merged_tail[-1] not in "。！？!?；;":
        merged_tail += "。"
    if merged_tail:
        kept.append(merged_tail)
    return "".join(part.strip() for part in kept if part).strip()


def _get_recent_assistant_text_from_conn(conn, limit: int = 3) -> str:
    if conn is None:
        return ""
    dialogue = getattr(getattr(conn, "dialogue", None), "dialogue", None)
    if not isinstance(dialogue, list):
        return ""

    pieces = []
    for message in reversed(dialogue):
        role = str(getattr(message, "role", "") or "").strip().lower()
        content = str(getattr(message, "content", "") or "").strip()
        if role != "assistant" or not content:
            continue
        pieces.append(content)
        if len(pieces) >= limit:
            break
    pieces.reverse()
    return normalize_spoken_text(" ".join(pieces))


def _get_recent_user_text_from_conn(conn, limit: int = 3) -> str:
    if conn is None:
        return ""
    dialogue = getattr(getattr(conn, "dialogue", None), "dialogue", None)
    if not isinstance(dialogue, list):
        return ""

    pieces = []
    for message in reversed(dialogue):
        role = str(getattr(message, "role", "") or "").strip().lower()
        content = str(getattr(message, "content", "") or "").strip()
        if role != "user" or not content:
            continue
        pieces.append(content)
        if len(pieces) >= limit:
            break
    pieces.reverse()
    return normalize_spoken_text(" ".join(pieces))


def _get_current_turn_server_mcp_tool_names(conn) -> list[str]:
    if conn is None:
        return []

    current_sentence_id = str(getattr(conn, "sentence_id", "") or "").strip()
    payload_sentence_id = str(
        getattr(conn, "_current_turn_server_mcp_sentence_id", "") or ""
    ).strip()
    if not current_sentence_id or current_sentence_id != payload_sentence_id:
        return []
    return list(getattr(conn, "_current_turn_server_mcp_tool_names", []) or [])


def _extract_experiment_step_snapshot(payload) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "", ""

    body = payload.get("result") if isinstance(payload.get("result"), dict) else payload

    step = body.get("step")
    if isinstance(step, dict):
        title = str(step.get("title", "") or "").strip()
        prompts = step.get("prompts") if isinstance(step.get("prompts"), dict) else {}
        instruction = str(prompts.get("instruction", "") or "").strip()
        return title, instruction

    summary = body.get("summary")
    if isinstance(summary, dict):
        current_step = (
            summary.get("current_step") if isinstance(summary.get("current_step"), dict) else {}
        )
        current_details = (
            summary.get("current_step_details")
            if isinstance(summary.get("current_step_details"), dict)
            else {}
        )
        title = str(current_step.get("title", "") or "").strip()
        instruction = str(current_details.get("instruction", "") or "").strip()
        return title, instruction

    state = body.get("state")
    if isinstance(state, dict):
        current_step = (
            state.get("current_step") if isinstance(state.get("current_step"), dict) else {}
        )
        title = str(current_step.get("title", "") or "").strip()
        instruction = str(current_step.get("instruction", "") or "").strip()
        return title, instruction

    return "", ""


def _compose_trusted_current_step_reply(conn) -> str:
    if conn is None:
        return ""

    title = ""
    instruction = ""
    for payload in (
        getattr(conn, "experiment_current_step", None),
        getattr(conn, "experiment_progress_summary", None),
    ):
        title, instruction = _extract_experiment_step_snapshot(payload)
        if title or instruction:
            break

    if not (title or instruction):
        step_id = str(getattr(conn, "experiment_current_step_id", "") or "").strip()
        yaml_steps = getattr(conn, "_experiment_yaml_steps_cache", None)
        if step_id and isinstance(yaml_steps, list):
            for step in yaml_steps:
                if not isinstance(step, dict):
                    continue
                candidate_step_id = str(step.get("id", "") or "").strip()
                if candidate_step_id != step_id:
                    continue
                title = str(step.get("title", "") or "").strip()
                prompts = step.get("prompts") if isinstance(step.get("prompts"), dict) else {}
                instruction = str(
                    prompts.get("instruction")
                    or step.get("instruction")
                    or step.get("description")
                    or ""
                ).strip()
                break

    parts = []
    if title:
        parts.append(f"现在做这一步：{title}")
    if instruction:
        clean_instruction = instruction.rstrip("。！？!?；; ").strip()
        if clean_instruction:
            parts.append(clean_instruction)

    if not parts:
        step_id = str(getattr(conn, "experiment_current_step_id", "") or "").strip()
        if not step_id:
            return ""
        parts.append("现在先按当前步骤继续。")

    reply = "，".join(part for part in parts if part).strip("， ")
    if reply and reply[-1] not in "。！？!?":
        reply += "。"
    return prepare_runtime_spoken_text(reply)


_EXPERIMENT_ALIGNMENT_TOOL_NAMES = {
    "create_session",
    "get_state",
    "get_step",
    "get_progress_summary",
    "get_current_progress",
    "start_trial",
    "add_field",
    "add_fields",
    "finish_trial",
    "can_proceed",
    "proceed_to_next_step",
    "redirect_to_step",
    "redo_trial",
    "modify_record",
    "uvvis_session",
    "uvvis_prepare_dark_current",
    "uvvis_measure_spectra",
    "uvvis_measure_kinetics",
    "uvvis_scan_start",
    "uvvis_scan_status",
    "uvvis_scan_result",
}


def _looks_like_experiment_control_turn(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    if not normalized:
        return False

    tokens = (
        "继续下一步",
        "下一步",
        "做好了",
        "做完了",
        "全部完成",
        "已经完成",
        "开始实验",
        "开始今天的实验",
        "准备好了",
        "可以拍照",
        "开始扫描",
        "可以开始扫描",
        "开始测量",
        "开始动力学",
        "丁达尔",
        "颜色稳定",
        "拍好了",
    )
    return any(token in normalized for token in tokens)


def _looks_like_experiment_step_or_scan_guidance(text: str) -> bool:
    normalized = normalize_spoken_text(text)
    if not normalized:
        return False
    if _looks_like_step_guidance_text(normalized):
        return True
    if _looks_like_reagent_addition_step_guidance(normalized):
        return True
    if _looks_like_observation_record_tail_guidance(normalized):
        return True

    guidance_tokens = (
        "开始扫描",
        "开始测量",
        "开始动力学",
        "丁达尔现象观察",
        "把环境调暗",
        "把一到五号样品位",
        "参比位",
        "样品位",
        "比色皿",
        "放好了告诉我",
        "看完后直接告诉我",
        "观察后告诉我",
    )
    return any(token in normalized for token in guidance_tokens)


def _conn_is_waiting_for_experiment_ready(conn) -> bool:
    last_text = _get_recent_assistant_text_from_conn(conn, limit=3)
    if not last_text:
        return False
    normalized = re.sub(r"\s+", "", last_text)
    tokens = (
        "准备好开始了吗",
        "准备好了吗",
        "可以开始了吗",
        "现在开始吗",
        "要开始了吗",
    )
    return any(token in normalized for token in tokens)


def stage_experiment_ready_guard_bypass_for_next_turn(conn, count: int = 1) -> None:
    if conn is None:
        return
    try:
        delta = int(count or 0)
    except (TypeError, ValueError):
        delta = 0
    if delta <= 0:
        delta = 1
    try:
        current = int(
            getattr(conn, "_experiment_ready_guard_bypass_pending_turns", 0) or 0
        )
    except (TypeError, ValueError):
        current = 0
    setattr(
        conn,
        "_experiment_ready_guard_bypass_pending_turns",
        max(0, current) + delta,
    )


def activate_experiment_ready_guard_bypass_for_current_sentence(
    conn,
    sentence_id: str = "",
    *,
    force: bool = False,
) -> bool:
    if conn is None:
        return False

    resolved_sentence_id = str(
        sentence_id or getattr(conn, "sentence_id", "") or ""
    ).strip()
    if not resolved_sentence_id:
        return False

    try:
        pending = int(
            getattr(conn, "_experiment_ready_guard_bypass_pending_turns", 0) or 0
        )
    except (TypeError, ValueError):
        pending = 0
    if not force and pending <= 0:
        return False

    setattr(
        conn,
        "_experiment_ready_guard_bypass_sentence_id",
        resolved_sentence_id,
    )
    if pending > 0:
        setattr(conn, "_experiment_ready_guard_bypass_pending_turns", pending - 1)
    return True


def _conn_has_experiment_ready_guard_bypass(conn) -> bool:
    if conn is None:
        return False

    current_sentence_id = str(getattr(conn, "sentence_id", "") or "").strip()
    bypass_sentence_id = str(
        getattr(conn, "_experiment_ready_guard_bypass_sentence_id", "") or ""
    ).strip()
    if current_sentence_id and bypass_sentence_id == current_sentence_id:
        return True

    raw_count = getattr(conn, "_experiment_ready_guard_bypass_count", 0)
    try:
        count = int(raw_count or 0)
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return False
    setattr(conn, "_experiment_ready_guard_bypass_count", count - 1)
    return True


def _looks_like_step_guidance_text(text: str) -> bool:
    normalized = normalize_spoken_text(text)
    if not normalized:
        return False
    guidance_tokens = (
        "现在做这一步",
        "接下来做这一步",
        "当前这一步",
        "做好后告诉我",
    )
    return any(token in normalized for token in guidance_tokens)


def _looks_like_reagent_addition_step_guidance(text: str) -> bool:
    normalized = normalize_spoken_text(text)
    if not normalized:
        return False

    subject_tokens = (
        "样品",
        "烧杯",
        "KBr",
        "溴化钾",
        "纯水",
        "NaBH4",
        "硼氢化钠",
        "AgNO3",
        "硝酸银",
        "H2O2",
        "过氧化氢",
        "柠檬酸钠",
    )
    if not any(token in normalized for token in subject_tokens):
        return False

    addition_tokens = ("加入", "滴加", "补加")
    if not any(token in normalized for token in addition_tokens):
        return False

    guidance_tokens = (
        "混匀",
        "搅拌",
        "计时",
        "观察",
        "颜色变化",
        "颜色稳定",
        "告诉我",
        "做好后",
    )
    return any(token in normalized for token in guidance_tokens)


def _looks_like_observation_record_tail_guidance(text: str) -> bool:
    normalized = normalize_spoken_text(text)
    if not normalized:
        return False

    subject_tokens = (
        "样品",
        "NaBH4",
        "硼氢化钠",
        "KBr",
        "溴化钾",
        "纯水",
    )
    if not any(token in normalized for token in subject_tokens):
        return False

    matched_groups = 0
    token_groups = (
        ("开始计时", "同时开始计时", "计时"),
        ("持续搅拌", "保持搅拌", "继续搅拌", "搅拌"),
        ("持续观察颜色变化", "观察颜色变化", "持续观察", "观察颜色"),
        ("等颜色稳定", "颜色稳定", "稳定后"),
        ("最终颜色", "几分钟", "告诉我"),
    )
    for group in token_groups:
        if any(token in normalized for token in group):
            matched_groups += 1
    return matched_groups >= 4


def _spoken_text_has_step_completion_prompt(text: str) -> bool:
    normalized = normalize_spoken_text(text)
    if not normalized:
        return False

    completion_tokens = (
        "做好后告诉我",
        "做好告诉我",
        "做好了告诉我",
        "做完告诉我",
        "完成后告诉我",
        "完成了告诉我",
        "做完了告诉我",
        "测完告诉我",
        "扫完告诉我",
        "放好后告诉我可以开始扫描",
        "放好后告诉我开始扫描",
        "告诉我可以开始扫描",
        "可以开始扫描",
        "结束后告诉我",
        "加完告诉我",
        "加好了告诉我",
        "拍完告诉我",
        "拍好了告诉我",
        "看完告诉我",
        "观察完告诉我",
        "记录完告诉我",
        "放好了告诉我",
        "开始时告诉我",
    )
    return any(token in normalized for token in completion_tokens)


def _looks_like_student_facing_scan_ready_sentence(text: str) -> bool:
    normalized = normalize_spoken_text(text)
    if not normalized:
        return False

    direct_ready_tokens = (
        "放好后告诉我可以开始扫描",
        "放好后告诉我开始扫描",
        "告诉我可以开始扫描",
        "告诉我开始扫描",
        "可以开始扫描时直接告诉我",
        "可以开始时直接说“开始扫描”",
        "可以开始时直接说开始扫描",
    )
    if any(token in normalized for token in direct_ready_tokens):
        return True

    if (
        "开始扫描" in normalized
        and "直接说" in normalized
        and "可以开始时" in normalized
    ):
        return True

    return "可以开始扫描" in normalized and any(
        token in normalized
        for token in ("请在", "放入", "放好", "样品位", "参比位", "比色皿")
    )


def _ensure_spoken_step_completion_prompt(text: str) -> str:
    normalized = normalize_spoken_text(text)
    if not normalized:
        return ""
    if _spoken_text_has_step_completion_prompt(normalized):
        return normalized

    looks_like_step_guidance = (
        _looks_like_step_guidance_text(normalized)
        or _looks_like_reagent_addition_step_guidance(normalized)
        or _looks_like_observation_record_tail_guidance(normalized)
    )
    if not looks_like_step_guidance:
        return normalized

    stripped = normalized.rstrip(" ，,、；;。！？!?")
    if not stripped:
        return normalized
    return f"{stripped}，做好后告诉我。"


def _spoken_text_has_measurement_detail(text: str) -> bool:
    normalized = normalize_spoken_text(text)
    if not normalized:
        return False

    primary_units = (
        "毫升",
        "微升",
        "升",
        "毫克",
        "克",
        "滴",
        "分钟",
        "秒",
    )
    if not any(unit in normalized for unit in primary_units):
        return False
    return bool(re.search(r"[零一二三四五六七八九十百千万两\d]", normalized))


def _spoken_text_has_reagent_amount_detail(text: str) -> bool:
    normalized = normalize_spoken_text(text)
    if not normalized:
        return False

    reagent_tokens = (
        "加入",
        "滴加",
        "补加",
        "KBr",
        "溴化钾",
        "纯水",
        "NaBH4",
        "硼氢化钠",
        "AgNO3",
        "硝酸银",
        "H2O2",
        "过氧化氢",
        "柠檬酸钠",
    )
    amount_units = (
        "毫升",
        "微升",
        "克",
        "毫克",
        "滴",
        "mL",
        "uL",
        "μL",
        "mg",
        "g",
    )
    has_numeric_amount = bool(
        re.search(r"[0-9零一二三四五六七八九十百千万点两]", normalized)
    )
    return (
        any(token in normalized for token in reagent_tokens)
        and any(unit in normalized for unit in amount_units)
        and has_numeric_amount
    )


def _compact_spoken_measurement_detail(detail: str) -> str:
    normalized = normalize_spoken_text(detail).strip(" ，,、；;")
    if not normalized:
        return ""

    if _spoken_text_has_measurement_detail(normalized):
        normalized = re.sub(
            r"(?:^|[\s，,、])(?:浓度)?[零一二三四五六七八九十百千万两\d\.点]+"
            r"(?:\s*[xX×]\s*10\s*[-−]?\s*\d+)?\s*"
            r"(?:摩尔每升|毫摩尔每升|微摩尔每升|纳摩尔每升|mol\s*/\s*L|mol/L|mM|μM|uM|"
            r"毫克每毫升|mg\s*/\s*mL|mg/mL|克每升|g\s*/\s*L|g/L|百分之[零一二三四五六七八九十百千万两\d\.]+|%)",
            " ",
            normalized,
        )

    parts = [part.strip() for part in re.split(r"[，,；;]\s*", normalized) if part.strip()]
    if not parts:
        parts = [normalized]

    cleaned_parts = []
    for part in parts:
        compact = re.sub(r"\s+", "", part).strip(" ，,、；;")
        if compact:
            cleaned_parts.append(compact)

    if not cleaned_parts:
        return ""
    if len(cleaned_parts) == 2:
        return "和".join(cleaned_parts)
    return "、".join(cleaned_parts)


def _compact_spoken_step_amount_parentheticals(text: str) -> str:
    if not text:
        return ""

    compacted = str(text)

    def _multi_addition_repl(match: re.Match) -> str:
        detail = _compact_spoken_measurement_detail(match.group("detail"))
        if not _spoken_text_has_measurement_detail(detail):
            return match.group(0)
        return f"加入{detail}"

    compacted = re.sub(
        r"(?P<head>(?:[一二三四五六七八九十0-9]+\s*号样品\s*)?"
        r"(?:[\u4e00-\u9fffA-Za-z0-9·/\-]+\s*(?:、|和|及|与)\s*)+"
        r"[\u4e00-\u9fffA-Za-z0-9·/\-]+)\s*加入\s*[（(](?P<detail>[^()（）]{1,120})[）)]",
        _multi_addition_repl,
        compacted,
    )

    def _single_addition_repl(match: re.Match) -> str:
        detail = _compact_spoken_measurement_detail(match.group("detail"))
        if not _spoken_text_has_measurement_detail(detail):
            return match.group(0)
        prefix = re.sub(r"\s+", "", match.group("prefix"))
        name = re.sub(r"\s+", "", match.group("name"))
        if detail.startswith(name):
            return f"{prefix}{detail}"
        return f"{prefix}{name}{detail}"

    compacted = re.sub(
        r"(?P<prefix>(?:先|再|快速|随后|继续|依次|分别)?\s*(?:加入|滴加))\s*"
        r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·/\-]+)\s*[（(](?P<detail>[^()（）]{1,120})[）)]",
        _single_addition_repl,
        compacted,
    )

    def _generic_amount_repl(match: re.Match) -> str:
        name = re.sub(r"\s+", "", match.group("name"))
        detail = _compact_spoken_measurement_detail(match.group("detail"))
        if not _spoken_text_has_measurement_detail(detail):
            return match.group(0)
        if detail.startswith(name):
            return detail
        return f"{name}{detail}"

    compacted = re.sub(
        r"(?P<name>[\u4e00-\u9fffA-Za-z0-9·/\-]+)\s*[（(](?P<detail>[^()（）]{1,120})[）)]",
        _generic_amount_repl,
        compacted,
    )
    return compacted


def _is_spoken_step_reporting_clause(text: str) -> bool:
    clause = normalize_spoken_text(text).strip(" ，,、；;。！？!?")
    if not clause:
        return False

    reporting_prefixes = (
        "记录",
        "记下",
        "记得记录",
        "填写",
        "填好",
        "补记",
        "汇报",
        "报告",
        "反馈",
        "同步记录",
        "再记录",
        "并记录",
        "然后记录",
    )
    if clause.startswith(reporting_prefixes):
        return True

    explicit_reporting_topics = (
        "收尾情况",
        "收尾状态",
        "记录字段",
        "记录结果",
        "结果填写",
    )
    if any(token in clause for token in explicit_reporting_topics):
        return True

    if "记录" in clause and any(token in clause for token in ("时间", "现象", "结果", "状态")):
        action_tokens = ("观察", "拍照", "测量", "加入", "滴加", "混匀", "搅拌", "静置")
        if not any(token in clause for token in action_tokens):
            return True
    return False


def _strip_spoken_step_reporting_clauses(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""

    clauses = [part.strip() for part in re.split(r"[，,；;]\s*", stripped) if part.strip()]
    kept_clauses = [
        clause for clause in clauses if not _is_spoken_step_reporting_clause(clause)
    ]
    if not kept_clauses:
        return ""
    if len(kept_clauses) == 1:
        return kept_clauses[0]
    return "，".join(kept_clauses)


def _rewrite_spoken_step_sequence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""

    match = re.match(
        r"^(?:完成|做完)\s*(?P<first>.+?)后(?:，|,)?\s*(?P<rest>.+)$",
        stripped,
    )
    if not match:
        return stripped

    first = match.group("first").strip(" ，,、；;")
    rest = match.group("rest").strip(" ，,、；;")
    if not first or not rest:
        return stripped
    return f"先{first}，再{rest}"


def _compact_spoken_step_guidance_sentence(sentence: str) -> str:
    stripped = str(sentence or "").strip()
    if not stripped:
        return ""

    terminal = stripped[-1] if stripped[-1] in "。！？!?；;" else ""
    body = stripped[:-1] if terminal else stripped
    body = _strip_spoken_step_reporting_clauses(body)
    body = _compact_spoken_step_amount_parentheticals(body)
    body = _rewrite_spoken_step_sequence(body)
    body = re.sub(r"注意继续保持搅拌", "注意持续搅拌", body)
    body = re.sub(r"继续保持搅拌", "持续搅拌", body)
    body = re.sub(r"并保持搅拌", "，持续搅拌", body)
    body = re.sub(r"保持搅拌", "持续搅拌", body)
    body = re.sub(r"\s+", " ", body).strip(" ，,、；;")
    body = re.sub(r"[，,、]{2,}", "，", body)
    body = re.sub(r"([：:])\s*([，,、])", r"\1", body)
    body = re.sub(r"[，,、]+\s*([。！？!?；;])", r"\1", body)
    if not body:
        return ""
    if terminal:
        body += terminal
    return body


def _merge_spoken_step_guidance_sentences(sentences: list[str]) -> list[str]:
    if len(sentences) < 2:
        return sentences

    first = sentences[0].strip()
    second = sentences[1].strip()
    first_match = re.match(
        r"^(?P<intro>(?:现在做这一步|接下来做这一步|当前这一步)[:：])(?P<body>.+?)[。！？!?；;]?$",
        first,
    )
    if not first_match or not _spoken_text_has_measurement_detail(second):
        return sentences
    if _spoken_text_has_reagent_amount_detail(first) and not _spoken_text_has_reagent_amount_detail(
        second
    ):
        return sentences

    first_body = first_match.group("body").strip()
    intro = first_match.group("intro")
    subject = ""
    if "：" in first_body:
        subject = first_body.split("：", 1)[0].strip()
    elif ":" in first_body:
        subject = first_body.split(":", 1)[0].strip()

    second_body = second.rstrip("。！？!?；;").strip()
    if subject:
        subject_pattern = re.sub(r"\s+", r"\\s*", re.escape(subject))
        second_body = re.sub(
            rf"^(?:先|再)?\s*{subject_pattern}\s*[:：]?\s*",
            lambda match: "先" if match.group(0).strip().startswith("先") else "",
            second_body,
        )
    second_body = re.sub(r"^完成\s*", "", second_body)
    second_body = re.sub(
        r"^(?P<first>.+?)后(?:，|,)?\s*(?P<rest>.+)$",
        lambda match: f"先{match.group('first').strip(' ，,、；;')}，再{match.group('rest').strip(' ，,、；;')}",
        second_body,
    )
    if subject and second_body and not second_body.startswith(subject):
        merged_body = f"{subject}：{second_body}"
    else:
        merged_body = second_body or first_body

    if not merged_body:
        return sentences

    merged_first = f"{intro}{merged_body}"
    if merged_first[-1] not in "。！？!?；;":
        merged_first += "。"

    if len(sentences) >= 3:
        safety = sentences[2].strip()
        safety_terminal = safety[-1] if safety and safety[-1] in "。！？!?；;" else ""
        safety_body = safety[:-1] if safety_terminal else safety
        if "持续搅拌" in merged_first:
            safety_body = _strip_spoken_step_reporting_clauses(safety_body)
            safety_clauses = [
                clause.strip()
                for clause in re.split(r"[，,；;]\s*", safety_body)
                if clause.strip()
            ]
            safety_clauses = [
                clause
                for clause in safety_clauses
                if not ("搅拌" in clause and ("持续" in clause or "保持" in clause))
            ]
            if safety_clauses:
                rewritten_safety = "，".join(safety_clauses)
                if safety_terminal:
                    rewritten_safety += safety_terminal
                sentences[2] = rewritten_safety

    return [merged_first] + sentences[2:]


def _compact_spoken_step_guidance_text(text: str) -> str:
    normalized = normalize_spoken_text(text)
    if not normalized or not _looks_like_step_guidance_text(normalized):
        return normalized

    sentences = _split_spoken_sentence_chunks(normalized)
    compacted_sentences = []
    for sentence in sentences:
        compacted = _compact_spoken_step_guidance_sentence(sentence)
        if compacted:
            compacted_sentences.append(compacted)

    compacted_sentences = _merge_spoken_step_guidance_sentences(compacted_sentences)
    return "".join(compacted_sentences).strip()


def _apply_experiment_ready_guard(conn, text: str) -> str:
    return text


def _apply_export_artifact_guard(conn, text: str) -> str:
    if not text or conn is None:
        return text

    guard = getattr(conn, "_pending_export_report_validation", None)
    if not isinstance(guard, dict) or not guard.get("active"):
        return text

    normalized = normalize_spoken_text(text)
    if not normalized:
        return normalized

    report_keywords = ("实验报告", "报告", "PDF", "pdf", "导出")
    if not any(keyword in normalized for keyword in report_keywords):
        return normalized

    all_expected_outputs_exist = bool(guard.get("all_expected_outputs_exist", False))
    if not all_expected_outputs_exist:
        return "实验报告还没有完整生成成功，请稍后再试。"
    return normalized


def _get_current_turn_server_mcp_payload(conn):
    if conn is None:
        return None

    current_sentence_id = str(getattr(conn, "sentence_id", "") or "").strip()
    payload_sentence_id = str(
        getattr(conn, "_last_server_mcp_sentence_id", "") or ""
    ).strip()
    if not current_sentence_id or current_sentence_id != payload_sentence_id:
        return None
    return getattr(conn, "_last_server_mcp_payload", None)


def _current_turn_had_experiment_graph_tool(conn) -> bool:
    if conn is None:
        return False

    current_sentence_id = str(getattr(conn, "sentence_id", "") or "").strip()
    tracked_sentence_id = str(
        getattr(conn, "_current_turn_server_mcp_sentence_id", "") or ""
    ).strip()
    if not current_sentence_id or current_sentence_id != tracked_sentence_id:
        return False

    graph_tools = {
        "create_session",
        "close_session",
        "get_state",
        "get_overview",
        "list_steps",
        "get_step",
        "get_schema",
        "get_progress_summary",
        "get_current_progress",
        "get_modifiable_records",
        "export_records",
        "export_records_to_yaml",
        "start_trial",
        "cancel_trial",
        "add_field",
        "add_fields",
        "finish_trial",
        "can_proceed",
        "proceed_to_next_step",
        "redirect_to_step",
        "redo_trial",
        "modify_record",
    }
    tool_names = getattr(conn, "_current_turn_server_mcp_tool_names", []) or []
    return any(str(tool_name or "").strip() in graph_tools for tool_name in tool_names)


def _payload_text_for_tool_failure_guard(payload) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip().lower()
    try:
        return json.dumps(payload, ensure_ascii=False).lower()
    except TypeError:
        return str(payload).strip().lower()


def _payload_text_looks_like_tool_failure(payload_text: str) -> bool:
    text = str(payload_text or "").strip().lower()
    if not text:
        return False

    failure_tokens = (
        '"success": false',
        '"ok": false',
        "error",
        "failed",
        "timeout",
        "busy",
        "occupied",
        "inaccessible",
        "not found",
        "missing",
        "exception",
        "disconnect",
        "占用",
        "超时",
        "失败",
        "未找到",
        "缺少",
        "没接通",
        "没连上",
    )
    return any(token in text for token in failure_tokens)


def _apply_tool_failure_narration_guard(conn, text: str) -> str:
    if not text or conn is None:
        return text

    normalized = normalize_spoken_text(text)
    if not normalized:
        return normalized

    current_turn_payload = _get_current_turn_server_mcp_payload(conn)
    current_turn_payload_text = _payload_text_for_tool_failure_guard(current_turn_payload)
    has_current_turn_tool_failure = _payload_text_looks_like_tool_failure(
        current_turn_payload_text
    )
    has_current_turn_busy_failure = has_current_turn_tool_failure and any(
        token in current_turn_payload_text
        for token in ("busy", "occupied", "inaccessible", "lease", "占用", "忙")
    )

    if not has_current_turn_tool_failure and any(
        token in normalized
        for token in (
            "实验图谱接口这轮没接通",
            "实验图谱接口没接通",
            "接口这轮没接通",
        )
    ):
        normalized = normalized.replace("实验图谱接口这轮没接通，", "")
        normalized = normalized.replace("实验图谱接口这轮没接通", "")
        normalized = normalized.replace("实验图谱接口没接通，", "")
        normalized = normalized.replace("实验图谱接口没接通", "")
        normalized = normalized.replace("接口这轮没接通，", "")
        normalized = normalized.replace("接口这轮没接通", "")
        normalized = normalize_spoken_text(normalized).strip("，,。；; ")

    uvvis_occupation_claim = any(
        token in normalized
        for token in (
            "被别的程序占用",
            "被其他程序占用",
            "暂时不能直接启动校正",
            "还没能直接启动校正",
        )
    )
    if uvvis_occupation_claim and not has_current_turn_busy_failure:
        return "UV-Vis 这边还没准备好，请稍后再试。"

    return normalize_spoken_text(normalized)


def _apply_experiment_graph_alignment_guard(conn, text: str) -> str:
    if not text or conn is None:
        return text

    normalized = normalize_spoken_text(text)
    if not normalized:
        return normalized

    session_id = str(getattr(conn, "experiment_session_id", "") or "").strip()
    current_step_id = str(getattr(conn, "experiment_current_step_id", "") or "").strip()
    if not session_id and not current_step_id:
        return normalized

    current_sentence_id = str(getattr(conn, "sentence_id", "") or "").strip()
    bypass_sentence_id = str(
        getattr(conn, "_experiment_ready_guard_bypass_sentence_id", "") or ""
    ).strip()
    if current_sentence_id and current_sentence_id == bypass_sentence_id:
        return normalized

    recent_user_text = _get_recent_user_text_from_conn(conn, limit=2)
    if not recent_user_text:
        return normalized
    if not _looks_like_experiment_control_turn(recent_user_text):
        return normalized

    if not _looks_like_experiment_step_or_scan_guidance(normalized):
        return normalized

    if _current_turn_had_experiment_graph_tool(conn):
        return normalized

    trusted_reply = _compose_trusted_current_step_reply(conn)
    return trusted_reply or normalized


def _strip_backstage_leading_clauses(text: str) -> str:
    cleaned = (text or "").strip()
    while cleaned:
        updated = cleaned
        for pattern in BACKSTAGE_LEADING_PATTERNS:
            updated = pattern.sub("", updated, count=1).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _looks_like_backstage_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if not any(keyword in stripped for keyword in BACKSTAGE_PROCESS_KEYWORDS):
        return False
    return stripped.startswith(BACKSTAGE_PREFIXES)


def _is_backstage_filler_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in BACKSTAGE_FILLER_PATTERNS)


def _rewrite_backstage_sentence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    for pattern, replacement in BACKSTAGE_REWRITE_PATTERNS:
        if pattern.fullmatch(stripped):
            return replacement
    return stripped


def _is_full_backstage_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in BACKSTAGE_FULL_SENTENCE_PATTERNS)


def _is_structural_backstage_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in STRUCTURAL_BACKSTAGE_PATTERNS)


def _is_transition_backstage_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in TRANSITION_BACKSTAGE_PATTERNS)


def filter_spoken_backstage_text(text):
    """Drop backend workflow narration while preserving complete student-facing instructions."""
    if not text:
        return text

    normalized = _collapse_duplicated_canonical_opening(text)
    if not normalized:
        return ""

    # Hard opening rule: if the canonical opening sentence exists, keep only that sentence.
    opening_match = CANONICAL_OPENING_RE.search(normalized)
    if opening_match:
        return opening_match.group(0)

    sentence_split_re = re.compile(r"(?<=[。！？!?；;])\s*")
    backstage_prefixes = (
        "我先",
        "我再",
        "我会",
        "我现在",
        "我继续",
        "先帮你",
        "接着",
        "然后",
        "马上",
        "正在",
        "会话已",
        "我这边",
    )
    backstage_keywords = (
        "后台",
        "会话",
        "读取",
        "读一下",
        "记录",
        "写入",
        "校验",
        "工具",
        "调用",
        "当前步骤",
        "当前子步骤",
        "切到下一步",
        "推进到下一步",
        "检索",
        "加载",
    )
    filler_re = re.compile(r"^(收到|好的|好|明白了?|嗯|继续)[，,。！？!?；; ]*$")

    def should_drop_sentence(sentence: str) -> bool:
        s = sentence.strip()
        if not s:
            return True
        if filler_re.fullmatch(s):
            return True
        if _looks_like_student_facing_scan_ready_sentence(s):
            return False
        if _is_full_backstage_sentence(s):
            return True
        if _is_structural_backstage_sentence(s):
            return True
        if _is_transition_backstage_sentence(s):
            return True

        has_backstage_keyword = any(k in s for k in backstage_keywords)
        starts_with_backstage_prefix = s.startswith(backstage_prefixes)
        if has_backstage_keyword and starts_with_backstage_prefix:
            return True

        # Also drop explicit backend narration even without the common prefixes.
        if "并先在后台" in s or "再读取当前步骤" in s or "读取实验概览" in s:
            return True
        return False

    kept_sentences = []
    for raw_sentence in sentence_split_re.split(normalized):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if should_drop_sentence(sentence):
            continue
        if not get_string_no_punctuation_or_emoji(sentence):
            continue
        if kept_sentences and kept_sentences[-1] == sentence:
            continue
        kept_sentences.append(sentence)

    if kept_sentences:
        return "".join(kept_sentences).strip()

    # Fallback: strip leading backstage clauses if possible; if the whole line is
    # backstage narration, return empty instead of replaying it.
    cleaned = _strip_backstage_leading_clauses(normalized).strip()
    if cleaned and cleaned != normalized:
        return filter_spoken_backstage_text(cleaned)

    rewritten = _rewrite_backstage_sentence(normalized).strip()
    if rewritten and rewritten != normalized:
        return filter_spoken_backstage_text(rewritten)

    if (
        _is_full_backstage_sentence(normalized)
        or _is_structural_backstage_sentence(normalized)
        or _is_transition_backstage_sentence(normalized)
        or _is_backstage_filler_sentence(normalized)
        or _looks_like_backstage_sentence(normalized)
    ):
        return ""

    return rewritten


def prepare_runtime_spoken_text(text):
    """Apply shared runtime speech policy before enqueueing any spoken reply."""
    normalized = normalize_spoken_text(text)
    if not normalized:
        return ""

    filtered = filter_spoken_backstage_text(normalized)
    filtered = _strip_spoken_meta_guidance_clauses(filtered)
    filtered = _strip_spoken_future_step_clauses(filtered)
    filtered = _strip_spoken_technical_details(filtered)
    filtered = _compact_spoken_step_guidance_text(filtered)
    filtered = _limit_spoken_sentence_count(filtered, max_sentences=2)
    filtered = _ensure_spoken_step_completion_prompt(filtered)
    return normalize_spoken_text(filtered)


def prepare_runtime_spoken_text_for_conn(conn, text):
    prepared = prepare_runtime_spoken_text(text)
    if not prepared:
        return ""
    prepared = _apply_experiment_ready_guard(conn, prepared)
    prepared = _apply_export_artifact_guard(conn, prepared)
    prepared = _apply_tool_failure_narration_guard(conn, prepared)
    prepared = _apply_experiment_graph_alignment_guard(conn, prepared)
    return normalize_spoken_text(prepared)
