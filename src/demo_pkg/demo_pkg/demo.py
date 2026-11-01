import rclpy
from rclpy.node import Node
from turtlesim.srv import TeleportAbsolute, SetPen
from std_srvs.srv import Empty
import json
import os
import threading
import time
from llama_cpp import Llama, LlamaGrammar

# GBNF
GRAMMAR_JSON = r'''
root    ::= "{" space "\"reasoning\"" space ":" space string space "," space "\"waypoints\"" space ":" space array space "}" space
array   ::= "[" space (point (space "," space point)*)? space "]"
point   ::= "{" space "\"x\"" space ":" space number space "," space "\"y\"" space ":" space number space "}"
penval ::= "\"up\"" | "\"down\""

number ::= [0-9]+ ("." [0-9]+)? | "-" [0-9]+ ("." [0-9]+)?
string ::= "\"" [^\"]* "\""
space   ::= [ \t\n\r]*
'''

class LlamaTurtlePainterNode(Node):
    def __init__(self, json_path="drawings.json"):
        super().__init__('llama_turtle_painter_node')
        
        # 1. テンプレート座標設定ファイル読込
        self.json_path = json_path
        self.templates = self.load_templates()
        
        # ROS2 サービス設定
        self.teleport_client = self.create_client(TeleportAbsolute, '/turtle1/teleport_absolute')
        self.set_pen_client = self.create_client(SetPen, '/turtle1/set_pen')
        self.clear_client = self.create_client(Empty, '/clear')
        
        # 2. llama.cpp モデル初期化
        model_path = "./models/Qwen3.5-9B-Q4_K_M.gguf" 
        self.get_logger().info(f"モデルをロード中: {model_path}")
        self.llm = Llama(model_path=model_path, n_ctx=3072, verbose=False)
        self.grammar = LlamaGrammar.from_string(GRAMMAR_JSON)
        
        self.get_logger().info(f'LLM イラスト作成ノード起動')

        # 3. ユーザー入力スレッド起動
        self.running = True
        self.input_thread = threading.Thread(target=self.interactive_loop)
        self.input_thread.daemon = True
        self.input_thread.start()

    def load_templates(self):
        """設定ファイル（JSON）から標準テンプレート座標をロード"""
        if not os.path.exists(self.json_path):
            self.get_logger().warn(f"設定ファイル '{self.json_path}' が見つかりません。")
            return {}
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f"テンプレートファイルの読込エラー: {e}")
            return {}

    def set_pen(self, r, g, b, width, off):
        """ペンの状態を変更するヘルパー関数"""
        while not self.set_pen_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('SetPen サービスを待機中...')
        req = SetPen.Request()
        req.r, req.g, req.b, req.width, req.off = r, g, b, width, off
        future = self.set_pen_client.call_async(req)
        #time.sleep(0.1)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

    def teleport_to(self, x, y, theta=0.0):
        """指定した座標にテレポート"""
        while not self.teleport_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Teleport サービスを待機中...')
        req = TeleportAbsolute.Request()
        req.x = float(x)
        req.y = float(y)
        req.theta = float(theta)
        
        future = self.teleport_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

    def clear_canvas(self):
        """キャンバス上の描画線を消去"""
        while not self.clear_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Clear サービスを待機中...')
        req = Empty.Request()
        future = self.clear_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

    def interactive_loop(self):
        #キャンバスをクリア
        self.clear_canvas()

        """イラストのお題の入力待ち"""
        print("\n=======================================================")
        print(" 描いてほしい単語（お題）を入力してください。")
        print("=======================================================\n")
        
        while self.running:
            try:
                word = input(">>> お題を入力: ").strip()
                if not word: continue
                if word.lower() == 'exit':
                    self.running = False
                    rclpy.shutdown()
                    break
                
                self.generate_and_paint(word)
                
            except (KeyboardInterrupt, EOFError):
                self.running = False
                rclpy.shutdown()
                break

    def generate_and_paint(self, word):
        """LLMに参照テンプレートを渡して座標を生成"""
        # テンプレート設定ファイルの内容を文字列化してシステムプロンプトに組み込む
        templates_str = json.dumps(self.templates, ensure_ascii=False, indent=2)
        
        system_prompt = (
            "あなたはROS2 turtlesimのキャンバス(0.0 から 11.0)の上にイラストを描く幾何学アーティストです。\n"
            "以下はあらかじめ定義された図形の標準参照座標テンプレートデータです:\n"
            f"```json\n{templates_str}\n```\n"
            "ユーザーのお題がテンプレートにある場合はその座標を参考にし、アレンジや新しいお題の場合はそれに応じて適切な通過座標リスト（waypoints）を生成してください。\n"
            "座標の範囲は必ず 2.0 〜 9.0 の間に収めてください。\n\n"
            "出力は必ず指定されたJSONフォーマットに準拠してください。"
        )
        
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n「{word}」を描いてください。<|im_end|>\n<|im_start|>assistant\n"
        
        print(f"LLMが描き方を考えています...")
        
        response = self.llm(
            prompt=full_prompt,
            max_tokens=1500,
            grammar=self.grammar,
            temperature=0.2
        )
        
        json_text = response['choices'][0]['text']
        
        try:
            data = json.loads(json_text)
            #print(f"LLMの考え: {data.get('reasoning')}")
            waypoints = data.get("waypoints", [])
            
            if not waypoints:
                print("座標リストが空です。")
                return

            #print(f"座標リスト (waypoints):")
            #print(json.dumps(waypoints, indent=2, ensure_ascii=False))

            print("キャンバスをクリアします...")
            self.clear_canvas()

            print(f"描画を開始します。総ステップ数: {len(waypoints)}")
            
            for i, pt in enumerate(waypoints):
                pen_state = str(pt.get('pen', 'down')).lower()

                # ペン設定オブジェクトの作成
                pen_req = SetPen.Request()
                pen_req.r = 255
                pen_req.g = 50
                pen_req.b = 50
                pen_req.width = 4

                # 最初の点、または "up" が指定されている場合は絶対にペンをオフ（off=1）にする
                if i == 0 or 'up' in pen_state:
                    pen_req.off = 1  # オフ
                else:
                    pen_req.off = 0  # オン

                # サービスを同期的に確実に実行
                future_pen = self.set_pen_client.call_async(pen_req)
                rclpy.spin_until_future_complete(self, future_pen, timeout_sec=0.5)

                # 念の為ペン状態変更後にわずかな余裕を持たせる
                time.sleep(0.05)

                # テレポート実行
                self.teleport_to(pt['x'], pt['y'])
                time.sleep(0.2)
                
            print("描き終わりました！\n")
            
        except Exception as e:
            print(f"描画エラー、または解析失敗: {e}")
            print(f"該当のLLM出力テキスト:\n{json_text}")

def main(args=None):
    rclpy.init(args=args)
    node = LlamaTurtlePainterNode()
    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()

