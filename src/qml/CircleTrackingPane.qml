import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import EBts 1.0

Rectangle {
    id: root
    color: "black"

    signal closed()
    signal rebuildBaseline()

    function setFrame(frame) {
        frameView.setFrame(frame)
    }

    Theme { id: theme }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            height: 30
            color: theme.surface

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 10
                anchors.rightMargin: 4

                Label {
                    text: "Circle Tracking"
                    color: theme.text
                    font.bold: true
                }
                Item { Layout.fillWidth: true }
                ToolButton {
                    text: "Rebuild Baseline"
                    ToolTip.visible: hovered
                    ToolTip.text: "Recollect the marker baseline; keep the sensor unloaded while it runs."
                    onClicked: root.rebuildBaseline()
                }
                ToolButton {
                    text: "✕"
                    onClicked: root.closed()
                }
            }
        }

        FrameView {
            id: frameView
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}
