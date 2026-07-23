import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12

// Console-only pane: no camera feed, just the log of what the shared
// recorder (manual ribbon button or externally-triggered start.cmd/stop.cmd)
// has done. See sequence_recording_controller.h for the underlying protocol.
Rectangle {
    id: root
    color: theme.background

    signal closed()

    Theme { id: theme }

    function appendLine(line) {
        logModel.append({ lineText: line })
        logView.positionViewAtEnd()
    }

    ListModel { id: logModel }

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
                    text: "Sequence Recording"
                    color: theme.text
                    font.bold: true
                }
                Item { Layout.fillWidth: true }
                ToolButton {
                    text: "✕"
                    onClicked: root.closed()
                }
            }
        }

        ListView {
            id: logView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: logModel
            delegate: Label {
                width: ListView.view.width
                text: lineText
                color: theme.mutedText
                font.family: "monospace"
                font.pixelSize: 12
                leftPadding: 10
                wrapMode: Text.WrapAnywhere
            }
        }
    }
}
