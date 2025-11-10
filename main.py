from view import TornadoAnalyzerGUI
from controller import TornadoController

def main():
    try:
        view = TornadoAnalyzerGUI(None)

        controller = TornadoController(view)
        view.controller = controller

        view.run()
        view.root.mainloop()
    except Exception as e:
        print(f"Erro ao iniciar aplicação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()